import random
import chess

PIECE_VALUES = {
    chess.PAWN: 157,
    chess.KNIGHT: 288,
    chess.BISHOP: 540,
    chess.ROOK: 605,
    chess.QUEEN: 561,
    chess.KING: 0,
}
CHECKMATE_SCORE = 100000


def evaluate_board(board: chess.Board, piece_values: dict) -> float:
    score = 0
    for piece_type, value in piece_values.items():
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


def move_priority(board: chess.Board, move: chess.Move, piece_values: dict) -> int:
    if not board.is_capture(move):
        return 0
    victim = board.piece_type_at(move.to_square) or chess.PAWN
    attacker = board.piece_type_at(move.from_square)
    return 10 * piece_values[victim] - piece_values[attacker]


def order_moves(board: chess.Board, piece_values: dict):
    moves = list(board.legal_moves)
    random.shuffle(moves)
    moves.sort(key=lambda m: move_priority(board, m, piece_values), reverse=True)
    return moves


def alpha_beta(
    board: chess.Board,
    depth: int,
    piece_values: dict,
    alpha: float = float('-inf'),
    beta: float = float('inf'),
) -> float:
    result = terminal_score(board)
    if result is not None:
        return result

    if depth <= 0:
        return evaluate_board(board, piece_values)

    if board.turn == chess.WHITE:
        best_score = float('-inf')
        for move in order_moves(board, piece_values):
            board.push(move)
            try:
                child_score = alpha_beta(board, depth - 1, piece_values, alpha, beta)
            finally:
                board.pop()
            best_score = max(best_score, child_score)
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return best_score

    else:
        best_score = float('inf')
        for move in order_moves(board, piece_values):
            board.push(move)
            try:
                child_score = alpha_beta(board, depth - 1, piece_values, alpha, beta)
            finally:
                board.pop()
            best_score = min(best_score, child_score)
            beta = min(beta, best_score)
            if alpha >= beta:
                break
        return best_score


def choose_move(board: chess.Board, depth: int = 10, piece_values: dict = PIECE_VALUES):
    if depth < 1:
        raise ValueError('Search depth must be at least 1.')
    if terminal_score(board) is not None:
        return None

    maximizing = board.turn == chess.WHITE
    best_score = float('-inf') if maximizing else float('inf')
    best_move = None
    alpha, beta = float('-inf'), float('inf')

    moves = order_moves(board, piece_values)

    for move in moves:
        board.push(move)
        try:
            score = alpha_beta(board, depth - 1, piece_values, alpha, beta)
        finally:
            board.pop()

        if best_move is None or (maximizing and score > best_score) or (
            not maximizing and score < best_score
        ):
            best_score = score
            best_move = move

        if maximizing:
            alpha = max(alpha, best_score)
        else:
            beta = min(beta, best_score)

    return best_move