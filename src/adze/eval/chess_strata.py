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
