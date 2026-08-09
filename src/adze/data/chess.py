"""Chess game generator for the Adze latent-diffusion experiment.

Each game becomes a sequence of plies (half-moves), one per block, analogous
to arithmetic `Trace`. Provenance tracking mirrors `Step.lhs_from / rhs_from`:

  lhs_from  — which ply last moved the piece being moved now, or None if the
               piece has not moved since the game's starting position.
  rhs_from  — for captures: which ply last moved the captured piece, or None if
               it has not moved. For castling: which ply last moved the rook.
               Always None for non-capture, non-castling moves.

The provenance formula `n = (lhs_from is not None) + (rhs_from is not None)`
maps to the same three class labels ("both-leaves" / "one-leaf" /
"both-from-earlier") used in the arithmetic strata analysis. The formula
transfers; the semantics do NOT. In arithmetic "both-from-earlier" means "the
prefix determines the step". In chess it means "a previously-moved piece captures
a previously-moved piece" — a mobility fact, not an information-theoretic one.
Read chess provenance classes as structural labels only, not as a carry-over of
the arithmetic interpretation.

Consumer distance: after ply k moves piece P to square S, the consumer of k is
the next ply that either moves P from S or captures P on S. Distance = consumer -
k, or None if P survives to the end of the game (the chess root).

## Data source

Use real games from Lichess monthly PGN dumps (https://database.lichess.org/).
Random-play generation is available for unit tests only — random moves have no
dependency structure, so the consumer-distance distribution is meaningless for
the measurement and the pin is unexploitable.

Filter Lichess dumps by time control to favour games where later moves are
genuinely conditioned on earlier ones: include classical (long base, no increment
required) and rapid (base ≥ 180s or increment ≥ 3s). Blitz/bullet games have
shorter dependency chains and more blunders; they should be excluded from
diagnostics and training.

## Promotion and castling

Promotion: the promoted piece gets a fresh identity distinct from the pawn,
keyed by the promotion ply index. The promotion ply's own `lhs_from` records
the pawn's prior history. The promoted piece's next move has `lhs_from` equal
to the promotion ply index.

Castling: treated as a genuine two-operand move. `lhs_from` = last ply that
moved the king (None if king on its starting square). `rhs_from` = last ply
that moved the castling rook (None if rook on its starting file). Both king and
rook are updated in `last_moved` after the castling ply.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn

# ── constants ──────────────────────────────────────────────────────────────────

PIECE_NAMES = {
    chess.PAWN:   "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK:   "R",
    chess.QUEEN:  "Q",
    chess.KING:   "K",
}


# ── data classes ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChessMove:
    """One ply in a chess game — structural analogue of arithmetic `Step`.

    Shapes / semantics:
        san                : Standard Algebraic Notation string, e.g. "Nf3",
                             "exd5", "O-O", "e8=Q".
        piece_type         : chess.PieceType of the piece doing the moving AFTER
                             the ply (promotions: queen's type, not the pawn's).
        from_sq            : 0-63 origin square.
        to_sq              : 0-63 destination square.
        is_capture         : True for captures and en-passant; False for quiet
                             moves and castling (castling uses rhs_from).
        captured_piece_type: chess.PieceType of the captured piece, or None.
                             For en-passant this is always PAWN. For castling
                             and quiet moves, always None.
        lhs_from           : ply index that last moved the moving piece, or None.
        rhs_from           : ply index that last moved the captured/rook piece,
                             or None.
    """

    san: str
    piece_type: int
    from_sq: int
    to_sq: int
    is_capture: bool
    captured_piece_type: int | None
    lhs_from: int | None
    rhs_from: int | None

    def render(self) -> str:
        """Return the SAN string — the text the tokeniser will see."""
        return self.san


@dataclass(frozen=True)
class ChessTrace:
    """A sequence of chess plies, structural analogue of arithmetic `Trace`.

    `game_id` is a short string identifying the source game (e.g. the Lichess
    game ID or a sequential index).
    """

    game_id: str
    moves: tuple[ChessMove, ...]

    def is_valid(self) -> bool:
        """Replay all moves with python-chess and verify each is legal."""
        board = chess.Board()
        for mv in self.moves:
            try:
                move = board.parse_san(mv.san)
            except (chess.IllegalMoveError, chess.InvalidMoveError,
                    chess.AmbiguousMoveError, ValueError):
                return False
            if move not in board.legal_moves:
                return False
            board.push(move)
        return True


# ── builder internals ─────────────────────────────────────────────────────────

def _piece_identity(color: chess.Color, piece_type: int, square: int) -> str:
    """Stable identifier for a piece, keyed to its starting square.

    Using the starting square rather than the current square means two pieces of
    the same type and colour remain distinguishable through the whole game. The
    key changes on promotion (see _game_to_moves).
    """
    c = "w" if color == chess.WHITE else "b"
    return f"{c}{PIECE_NAMES[piece_type]}_{chess.square_name(square)}"


def _game_to_moves(game: chess.pgn.Game) -> list[ChessMove] | None:
    """Parse a PGN game into a list of ChessMove objects.

    Returns None if the game has no moves, ends in an error, or contains a
    position python-chess cannot validate.
    """
    board = chess.Board()
    moves: list[ChessMove] = []

    # sq_to_piece: current board square → piece identity string
    # last_moved:  piece identity → ply index of its last move
    sq_to_piece: dict[int, str] = {}
    last_moved: dict[str, int] = {}

    # Populate starting positions.
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            sq_to_piece[sq] = _piece_identity(piece.color, piece.piece_type, sq)

    node = game
    while node.variations:
        node = node.variations[0]
        mv = node.move
        if mv is None:
            return None

        try:
            san = board.san(mv)
        except Exception:
            return None

        from_sq = mv.from_square
        to_sq = mv.to_square
        moving_piece = board.piece_at(from_sq)
        if moving_piece is None:
            return None

        moving_id = sq_to_piece.get(from_sq)
        if moving_id is None:
            return None

        is_castling = board.is_castling(mv)
        is_capture = board.is_capture(mv)
        is_ep = board.is_en_passant(mv)

        # --- rhs_from and captured_piece_type: captured piece / rook ---
        rhs_from: int | None = None
        captured_piece_type: int | None = None

        if is_castling:
            # Identify the rook being castled with.
            if mv.to_square > mv.from_square:
                rook_sq = chess.H1 if moving_piece.color == chess.WHITE else chess.H8
            else:
                rook_sq = chess.A1 if moving_piece.color == chess.WHITE else chess.A8
            rook_id = sq_to_piece.get(rook_sq)
            if rook_id is not None:
                rhs_from = last_moved.get(rook_id)
            # captured_piece_type stays None — castling is not a capture

        elif is_ep:
            # En-passant: the captured pawn is on a different square from to_sq.
            ep_sq = chess.square(chess.square_file(to_sq),
                                 chess.square_rank(from_sq))
            cap_id = sq_to_piece.get(ep_sq)
            if cap_id is not None:
                rhs_from = last_moved.get(cap_id)
                sq_to_piece.pop(ep_sq, None)
            captured_piece_type = chess.PAWN  # always a pawn in en-passant

        elif is_capture:
            cap_id = sq_to_piece.get(to_sq)
            if cap_id is not None:
                rhs_from = last_moved.get(cap_id)
            cap_piece = board.piece_at(to_sq)
            if cap_piece is not None:
                captured_piece_type = cap_piece.piece_type

        # --- lhs_from: moving piece ---
        lhs_from: int | None = last_moved.get(moving_id)

        # --- determine piece type after the move (promotion changes it) ---
        if mv.promotion is not None:
            # The pawn promotes.  The promotion ply's piece_type is the NEW piece.
            piece_type_after = mv.promotion
            # New identity for the promoted piece.
            promo_id = f"{moving_id[0]}{PIECE_NAMES[piece_type_after]}_promo{len(moves)}"
            # Remove the pawn from tracking.
            sq_to_piece.pop(from_sq, None)
            last_moved.pop(moving_id, None)
            # Register the new piece at its destination, recording this ply as
            # the one that placed it.
            last_moved[promo_id] = len(moves)
            sq_to_piece[to_sq] = promo_id
        else:
            piece_type_after = moving_piece.piece_type
            # Move the piece to its destination.
            sq_to_piece.pop(from_sq, None)
            if is_capture and not is_ep:
                sq_to_piece.pop(to_sq, None)
            last_moved[moving_id] = len(moves)
            sq_to_piece[to_sq] = moving_id

        # For castling, also update the rook.
        if is_castling:
            if mv.to_square > mv.from_square:
                rook_from = chess.H1 if moving_piece.color == chess.WHITE else chess.H8
                rook_to = chess.F1 if moving_piece.color == chess.WHITE else chess.F8
            else:
                rook_from = chess.A1 if moving_piece.color == chess.WHITE else chess.A8
                rook_to = chess.D1 if moving_piece.color == chess.WHITE else chess.D8
            rook_id2 = sq_to_piece.pop(rook_from, None)
            if rook_id2 is not None:
                last_moved[rook_id2] = len(moves)
                sq_to_piece[rook_to] = rook_id2

        moves.append(ChessMove(
            san=san,
            piece_type=piece_type_after,
            from_sq=from_sq,
            to_sq=to_sq,
            is_capture=is_capture or is_ep,
            captured_piece_type=captured_piece_type,
            lhs_from=lhs_from,
            rhs_from=rhs_from,
        ))

        try:
            board.push(mv)
        except Exception:
            return None

    return moves if moves else None


# ── public API ─────────────────────────────────────────────────────────────────

def game_to_trace(game: chess.pgn.Game,
                  game_id: str) -> ChessTrace | None:
    """Convert a python-chess Game to a ChessTrace.

    Returns None if the game has no moves or contains an error.
    """
    moves = _game_to_moves(game)
    if moves is None:
        return None
    return ChessTrace(game_id=game_id, moves=tuple(moves))


def _is_classical_or_rapid(tc_str: str | None) -> bool:
    """Return True for time controls worth including in the diagnostic.

    Acceptable:
      - Classical / Correspondence: base >= 600s (10 min+)
      - Rapid: base >= 180s (3 min+) OR increment >= 3s
    Rejected:
      - Blitz: base < 180s, increment < 3
      - Bullet / UltraBullet
      - '-' (no time control recorded)
      - '?' (unknown)

    The filter is intentionally conservative: dependency chains in faster games
    are shorter and moves are more random, weakening the "later moves conditioned
    on earlier ones" premise.
    """
    if not tc_str or tc_str in ("-", "?"):
        return False
    # Increment format: "base+inc" (both in seconds) or just "base"
    try:
        if "+" in tc_str:
            base_s, inc_s = tc_str.split("+")
            base = int(base_s)
            inc = int(inc_s)
        else:
            base = int(tc_str)
            inc = 0
    except ValueError:
        return False
    return base >= 180 or inc >= 3


def load_pgn(path: Path,
             n: int = 1000,
             filter_time_control: bool = True) -> list[ChessTrace]:
    """Load up to `n` games from a PGN file, building a ChessTrace per game.

    Args:
        path: Path to a PGN file (plain text or .pgn; NOT compressed —
              decompress a Lichess .zst file with `zstd -d` first).
        n: Maximum number of traces to return.
        filter_time_control: If True, skip blitz/bullet games; keep
            classical and rapid only. Set False to load any game.

    Returns:
        List of ChessTrace objects, length ≤ n.
    """
    traces: list[ChessTrace] = []
    idx = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        while len(traces) < n:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            if filter_time_control:
                tc = game.headers.get("TimeControl")
                if not _is_classical_or_rapid(tc):
                    continue
            trace = game_to_trace(game, game_id=str(idx))
            idx += 1
            if trace is not None:
                traces.append(trace)
    return traces


def generate_random_trace(seed: int, max_plies: int = 80) -> ChessTrace:
    """Play random legal moves and return a ChessTrace.

    FOR UNIT TESTS ONLY. Random play has no dependency structure — the
    consumer-distance distribution carries no information about real game
    properties. Do not use this for diagnostics or training data.
    """
    rng = random.Random(seed)
    board = chess.Board()
    game = chess.pgn.Game()
    node: chess.pgn.GameNode = game

    for _ in range(max_plies):
        legal = list(board.legal_moves)
        if not legal:
            break
        mv = rng.choice(legal)
        node = node.add_variation(mv)
        board.push(mv)
        if board.is_game_over():
            break

    trace = game_to_trace(game, game_id=f"random_{seed}")
    if trace is None:
        # Fallback: an empty game — should not happen with a valid board.
        return ChessTrace(game_id=f"random_{seed}", moves=())
    return trace


def generate_random_dataset(n: int,
                             seed: int = 0,
                             max_plies: int = 80) -> list[ChessTrace]:
    """Generate n random-play chess traces.

    FOR UNIT TESTS ONLY. See `generate_random_trace` for the warning.
    Trace i is seeded from `seed * 1_000_003 + i`, matching the arithmetic
    generator's convention.
    """
    return [generate_random_trace(seed * 1_000_003 + i, max_plies)
            for i in range(n)]
