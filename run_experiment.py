"""Run 20 games (10 positions, both colors) between learning_algorithm and base_algorithm.

Pass a custom piece_values dict to run_experiment(piece_values) to evaluate any mutation
and append the result to results/experiments_dataset.csv.
"""
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import chess
import base_algorithm
import learning_algorithm
from learning_algorithm import PIECE_VALUES

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.json"
CANDIDATE = "learning_algorithm.py"
BASELINE = "base_algorithm.py"
FIRST_SEED = 1000  # Game n uses seed FIRST_SEED + n - 1.

POSITIONS = [
    ("Start position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("Italian Game", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("Ruy Lopez", "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("Sicilian Defence", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    ("French Defence", "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    ("Caro-Kann Defence", "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    ("Scandinavian Defence", "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    ("Queen's Gambit", "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2"),
    ("King's Indian Defence", "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"),
    ("English Opening", "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1"),
]
WHITE_POINTS = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def can_claim_draw(board: chess.Board, auto_claim: bool) -> str | None:
    if not auto_claim:
        return None
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.termination == chess.Termination.FIFTY_MOVES:
        condition = board.is_fifty_moves
        label = "50-move rule"
    elif outcome.termination == chess.Termination.THREEFOLD_REPETITION:
        condition = lambda: board.is_repetition(3)
        label = "Threefold repetition"
    else:
        return None

    if condition():
        return f"Draw claimed: {label}"
    for move in list(board.legal_moves):
        board.push(move)
        try:
            if condition():
                return f"Draw claimed: {label}"
        finally:
            board.pop()
    return None


def play_game(fen: str, depth: int, candidate_color: str, seed: int, auto_claim: bool, piece_values: dict):
    random.seed(seed)
    board = chess.Board(fen)
    candidate_turn = chess.WHITE if candidate_color == "white" else chess.BLACK

    while not board.is_game_over(claim_draw=False):
        claim_reason = can_claim_draw(board, auto_claim)
        if claim_reason:
            return "1/2-1/2", claim_reason

        board_copy = board.copy(stack=True)
        if board.turn == candidate_turn:
            move = learning_algorithm.choose_move(board_copy, depth, piece_values)
        else:
            move = base_algorithm.choose_move(board_copy, depth)

        if not isinstance(move, chess.Move) or move not in board.legal_moves:
            return "error", "Illegal move returned"
        board.push(move)

    outcome = board.outcome(claim_draw=False)
    if outcome.winner is None:
        return "1/2-1/2", f"Draw: {outcome.termination.name.lower()}"
    winner = "White" if outcome.winner == chess.WHITE else "Black"
    return board.result(), f"{winner} wins: Checkmate"


def append_experiment_summary(rows, depth: int, piece_values: dict):
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    log_file = out / "experiments_dataset.csv"

    scores = [r["candidate_score"] for r in rows if r["candidate_score"] != ""]
    total_games = len(scores)
    cand_score = sum(scores)
    wins = scores.count(1.0)
    draws = scores.count(0.5)
    losses = scores.count(0.0)
    win_rate = round(cand_score / total_games, 4) if total_games > 0 else 0.0

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pawn": piece_values[chess.PAWN],
        "knight": piece_values[chess.KNIGHT],
        "bishop": piece_values[chess.BISHOP],
        "rook": piece_values[chess.ROOK],
        "queen": piece_values[chess.QUEEN],
        "score": cand_score,
        "total_games": total_games,
        "win_rate": win_rate,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "depth": depth,
        "first_seed": FIRST_SEED,
        "candidate": CANDIDATE,
        "baseline": BASELINE,
    }

    file_exists = log_file.exists() and log_file.stat().st_size > 0
    fieldnames = list(entry.keys())

    if file_exists:
        with open(log_file, "rb") as f:
            f.seek(-1, 2)
            if f.read(1) not in (b"\n", b"\r"):
                with open(log_file, "a") as af:
                    af.write("\n")

    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)

    print("\n" + "=" * 60)
    print(f"Experiment finished (depth={depth}): Score {cand_score}/{total_games} (W {wins}, D {draws}, L {losses})")
    print(f"Piece values tested: P={piece_values[chess.PAWN]}, N={piece_values[chess.KNIGHT]}, "
          f"B={piece_values[chess.BISHOP]}, R={piece_values[chess.ROOK]}, Q={piece_values[chess.QUEEN]}")
    print(f"Appended result to {log_file.relative_to(ROOT)}")
    print("=" * 60 + "\n")
    return cand_score


def run_experiment(piece_values: dict = PIECE_VALUES, depth: int = None):
    config = json.loads(CONFIG.read_text())
    if depth is None:
        depth = config.get("search_depth", 4)
    auto_claim = config.get("rules", {}).get("auto_claim_draws_for_ai", True)
    rows = []

    for name, fen in POSITIONS:
        for candidate_color in ("white", "black"):
            game = len(rows) + 1
            seed = FIRST_SEED + game - 1
            white, black = ((CANDIDATE, BASELINE) if candidate_color == "white"
                            else (BASELINE, CANDIDATE))

            print(f"Game {game}/{2 * len(POSITIONS)}: {name}, candidate is {candidate_color}, seed {seed}")
            result, reason = play_game(fen, depth, candidate_color, seed, auto_claim, piece_values)

            score = ""
            if result in WHITE_POINTS:
                score = WHITE_POINTS[result] if candidate_color == "white" else 1 - WHITE_POINTS[result]
            print(f"  -> {result} ({reason}), candidate score: {score}")
            rows.append(dict(game=game, position=name, candidate_color=candidate_color,
                             white=white, black=black, seed=seed, result=result,
                             candidate_score=score, reason=reason, fen=fen))

    return append_experiment_summary(rows, depth, piece_values)


if __name__ == "__main__":
    run_experiment()
