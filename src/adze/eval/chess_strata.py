"""Stratification helpers for the chess domain.

These mirror `adze.eval.strata.consumer_distance` and
`adze.eval.strata.operand_provenance`, but operate on `ChessTrace` /
`ChessMove` objects.

## On provenance semantics

The arithmetic provenance classes ("both-leaves", "one-leaf",
"both-from-earlier") were interpreted as an information-theoretic claim: the
prefix determines the step to varying degrees. In chess, the same formula —
n = (lhs_from is not None) + (rhs_from is not None) — produces the same three
labels, but they carry a MOBILITY meaning, not an information-theoretic one.

  "both-from-earlier" in chess  =  a previously-moved piece captures a
                                    previously-moved piece

That says nothing about whether the prefix determines the move. A queen that
has moved 10 times capturing another well-travelled queen is classified
"both-from-earlier", but the move may be forced (determined by the position)
or arbitrary (many captures equally legal).

**Do not read chess results through the arithmetic story.** Report them as
structural labels that are parallel in form and distinct in content.

## Piece-type confound

Consumer distance is confounded by piece mobility independently of provenance.
Pawns move infrequently and are often the last piece to touch a square, giving
long consumer distances by mobility alone. Queens move frequently, giving short
consumer distances. Since "both-from-earlier" correlates with high-mobility
pieces (they move more, so they are more likely to have moved before), the
provenance–distance coupling in chess may be driven by piece type rather than
dependency structure.

Always report piece-type composition alongside provenance–distance cross-tabs,
and verify the coupling survives stratifying by piece type before drawing
conclusions about the dependency structure.
"""

from __future__ import annotations

import chess

from adze.data.chess import ChessMove, ChessTrace
from adze.eval.strata import PROVENANCE

PIECE_TYPE_NAMES = {
    chess.PAWN:   "PAWN",
    chess.KNIGHT: "KNIGHT",
    chess.BISHOP: "BISHOP",
    chess.ROOK:   "ROOK",
    chess.QUEEN:  "QUEEN",
    chess.KING:   "KING",
}


def chess_consumer_distance(trace: ChessTrace, move_idx: int) -> int | None:
    """Plies from `move_idx` to the next ply that moves or captures that piece.

    After ply k places piece P on square S, the consumer of k is the next ply
    j > k where:
      - P is the moving piece (moved from S), or
      - P is captured at S.

    Returns None if P survives to the end of the game (the chess root analogue).
    """
    if move_idx < 0 or move_idx >= len(trace.moves):
        raise IndexError(f"move_idx {move_idx} out of range for trace of length {len(trace.moves)}")

    mv = trace.moves[move_idx]
    landing_sq = mv.to_sq

    # Identify the piece that landed on landing_sq at move_idx.
    # We need to track where it goes from there. Walk forward.
    # The piece's position changes when it next moves; another piece may
    # occupy landing_sq after a capture.

    # We only need to track the specific piece placed at move_idx.
    # Strategy: replay the squares. Keep track of "which piece is on landing_sq
    # right now" by scanning forward for any move with from_sq == landing_sq
    # (piece leaves) or to_sq == landing_sq (another piece arrives, potentially
    # capturing ours).
    #
    # A piece is consumed when:
    #   - it moves away (from_sq == current_sq), OR
    #   - it is captured (to_sq == current_sq AND is_capture)
    #
    # En-passant is handled in chess.py: is_capture is True for en-passant and
    # the pawn leaves its column (from_sq != to_sq), but the *captured* pawn's
    # square is adjacent. We cannot determine en-passant from SAN alone here.
    # Accept a small inaccuracy: en-passant captures record is_capture=True but
    # the pawn on the captured rank is not at to_sq. In practice en-passant is
    # rare and the error is small.

    current_sq = landing_sq

    for j in range(move_idx + 1, len(trace.moves)):
        nxt = trace.moves[j]
        if nxt.from_sq == current_sq:
            # Our piece moved; this ply is the consumer.
            return j - move_idx
        if nxt.to_sq == current_sq and nxt.is_capture:
            # Our piece was captured here.
            return j - move_idx

    return None


def chess_occupancy_consumer_map(
    trace: ChessTrace,
) -> dict[int, tuple[int, ...]]:
    """Definition C: occupancy-based consumer map.

    Move j is a consumer of move i if move i placed a piece on a square that
    j's legality depends on.  Three kinds of dependency square are tracked:

      1. j's destination (to_sq) — a piece placed there by i that j captures.
      2. Sliding path squares (empty at time j) — the ply that last vacated each
         intermediate square enabled j's sliding move through that square.
      3. Check/pin squares — the ply that placed the checking or pinning piece.

    Returns:
        consumer_map[i] = sorted tuple of j > i that depend on i's placement or
        clearance.

    Call once per trace; result is O(N^2) in the worst case for multi-consumer
    counts but is single-pass over moves with O(64) inner work per ply.
    """
    board = chess.Board()
    n = len(trace.moves)

    # sq_to_placer[sq]  = ply index that last PLACED a piece on sq.
    #                      None  →  originally occupied (not from any ply).
    # sq_to_vacater[sq] = ply index of the move that last moved a piece OFF sq.
    #                      None  →  originally empty or never vacated by a ply.
    sq_to_placer:  dict[int, int | None] = {}
    sq_to_vacater: dict[int, int | None] = {}

    for sq in chess.SQUARES:
        if board.piece_at(sq) is not None:
            sq_to_placer[sq] = None   # original piece, no ply placed it
        else:
            sq_to_vacater[sq] = None  # originally empty, no ply vacated it

    # producers[i] = set of j's that depend on i
    producers: dict[int, set[int]] = {i: set() for i in range(n)}

    def _record(dep_sq: int, j: int, *, from_placer: bool) -> None:
        """Add edge i → j if dep_sq was touched by a non-None ply."""
        if from_placer:
            i = sq_to_placer.get(dep_sq)
        else:
            i = sq_to_vacater.get(dep_sq)
        if i is not None:
            producers[i].add(j)

    for j in range(n):
        mv_obj = trace.moves[j]
        try:
            mv = board.parse_san(mv_obj.san)
        except Exception:
            try:
                board.push_san(mv_obj.san)
            except Exception:
                pass
            continue

        pt = board.piece_type_at(mv.from_square)

        # 1. Destination square — the piece there (if any) was placed by some i.
        _record(mv.to_square, j, from_placer=True)

        # 2. Sliding path — intermediate squares are empty for a legal move;
        #    the ply that last vacated each enables j's sliding line.
        if pt in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            for sq in chess.SquareSet(chess.between(mv.from_square, mv.to_square)):
                _record(sq, j, from_placer=False)

        # 3. Checker squares — the piece giving check was placed by some i.
        for sq in board.checkers():
            _record(sq, j, from_placer=True)

        # 4. Pinning piece — the pinner was placed by some i.
        pin_mask = board.pin(board.turn, mv.from_square)
        if pin_mask != chess.BB_ALL:
            king_sq = board.king(board.turn)
            for sq in chess.SquareSet(pin_mask):
                if sq == mv.from_square or sq == king_sq:
                    continue
                p = board.piece_at(sq)
                if p is not None and p.color != board.turn:
                    _record(sq, j, from_placer=True)
                    break

        # ── update tables ────────────────────────────────────────────────────
        # vacate from_square
        sq_to_placer.pop(mv.from_square, None)
        sq_to_vacater[mv.from_square] = j

        # handle en passant: captured pawn on ep_sq
        if board.is_en_passant(mv):
            ep_sq = chess.square(chess.square_file(mv.to_square),
                                 chess.square_rank(mv.from_square))
            sq_to_placer.pop(ep_sq, None)
            sq_to_vacater[ep_sq] = j
        elif board.is_capture(mv):
            # captured piece removed from to_sq before placing the moving piece
            sq_to_placer.pop(mv.to_square, None)
            # to_sq is about to be filled — do NOT mark it as vacated

        # place moving piece at to_sq
        sq_to_placer[mv.to_square] = j
        sq_to_vacater.pop(mv.to_square, None)   # sq is now occupied, not empty

        # handle castling: rook also moves
        if board.is_castling(mv):
            if mv.to_square > mv.from_square:   # kingside
                rook_from = chess.H1 if board.turn == chess.WHITE else chess.H8
                rook_to   = chess.F1 if board.turn == chess.WHITE else chess.F8
            else:                                # queenside
                rook_from = chess.A1 if board.turn == chess.WHITE else chess.A8
                rook_to   = chess.D1 if board.turn == chess.WHITE else chess.D8
            sq_to_placer.pop(rook_from, None)
            sq_to_vacater[rook_from] = j
            sq_to_placer[rook_to] = j
            sq_to_vacater.pop(rook_to, None)

        board.push(mv)

    return {i: tuple(sorted(producers[i])) for i in range(n)}


def chess_operand_provenance(move: ChessMove) -> str:
    """Structural provenance class of a chess ply.

    Uses the same formula as arithmetic operand_provenance:
        n = (lhs_from is not None) + (rhs_from is not None)
    mapping to "both-leaves" (0), "one-leaf" (1), "both-from-earlier" (2).

    The classes are structurally parallel to arithmetic and semantically
    distinct — see module docstring before interpreting results.
    """
    n = (move.lhs_from is not None) + (move.rhs_from is not None)
    return PROVENANCE[n]
