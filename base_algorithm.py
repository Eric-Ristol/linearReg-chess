import random
import chess


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
CHECKMATE_SCORE = 100000


def evaluate_board(board: chess.Board) -> float:
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        score += value * (white_count - black_count)
    return score

def terminal_score(board: chess.Board):
    outcome = board.outcome(claim_draw=False)
    if outcome is None:
        return None
    if outcome.winner is chess.WHITE:
        return CHECKMATE_SCORE
    if outcome.winner is chess.BLACK:
        return -CHECKMATE_SCORE
    return 0

def move_priority(board: chess.Board, move: chess.Move) -> int:
    """Captures first: most valuable victim, then least valuable attacker."""
    if not board.is_capture(move):
        return 0
    victim = board.piece_type_at(move.to_square) or chess.PAWN  # en passant
    attacker = board.piece_type_at(move.from_square)
    return 10 * PIECE_VALUES[victim] - PIECE_VALUES[attacker]

def order_moves(board: chess.Board):
    """
    Order moves to improve alpha-beta pruning efficiency. 
    Moves are shuffled first to vary equally good choices, then sorted 
    by MVV-LVA (most valuable victim, least valuable attacker).
    """
    moves = list(board.legal_moves)
    random.shuffle(moves)
    moves.sort(key=lambda m: move_priority(board, m), reverse=True)
    return moves

def alpha_beta(
    board: chess.Board,
    depth: int,
    alpha: float = float('-inf'),
    beta: float = float('inf'),
) -> float:
    """
    Search a position with the bounds inherited from its ancestors.
    alpha: the lower bound established by White's alternatives so far.
    beta: the upper bound established by Black's alternatives so far.
    When alpha >= beta, remaining siblings cannot improve an ancestor's
    decision, so they can be skipped. A full window (-inf, inf) returns the
    exact minimax score; a cutoff may return only a bound on that score.
    """
    # Check endings before the depth limit, including mates at the horizon.
    result = terminal_score(board)
    if result is not None:
        return result
    
    if depth <= 0:
        return evaluate_board(board)

    if board.turn == chess.WHITE:
        best_score = float('-inf')
        for move in order_moves(board):
            board.push(move)
            try:
                child_score = alpha_beta(board, depth - 1, alpha, beta)
            finally:
                board.pop()
            best_score = max(best_score, child_score)
            # White can already obtain at least this score.
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break  # Black has an alternative that is no worse.
        return best_score

    else:
        best_score = float('inf')
        for move in order_moves(board):
            board.push(move)
            try:
                child_score = alpha_beta(board, depth - 1, alpha, beta)
            finally:
                board.pop()
            best_score = min(best_score, child_score)
            # Black can already hold the score down to this value.
            beta = min(beta, best_score)
            if alpha >= beta:
                break  # White has an alternative that is no worse.
        return best_score

def choose_move(board: chess.Board, depth: int = 10):

    """Return a legal move, or None for a finished game, without changing board."""
    if depth < 1:
        raise ValueError('Search depth must be at least 1.')
    if terminal_score(board) is not None:
        return None

    maximizing = board.turn == chess.WHITE
    best_score = float('-inf') if maximizing else float('inf')
    best_move = None
    alpha, beta = float('-inf'), float('inf')

    # Use the custom order_moves function for the root as well.
    moves = order_moves(board)

    for move in moves:
        board.push(move)
        try:
            score = alpha_beta(board, depth - 1, alpha, beta)
        finally:
            board.pop()
            
        # Keep the first move on a tie. Internal cutoff scores can be bounds,
        # so a later equal return is not necessarily an exact minimax tie.
        if best_move is None or (maximizing and score > best_score) or (
            not maximizing and score < best_score
        ):
            best_score = score
            best_move = move

        # Carry the root's best bound into the next sibling's search.
        # Resetting both bounds for every move would lose some pruning.
        if maximizing:
            alpha = max(alpha, best_score)
        else:
            beta = min(beta, best_score)
            
    return best_move