from pathlib import Path
import random
import chess
from main import main
from run_experiment import run_experiment
from ml_functions import generation_learning

GENERATIONS = 2
EXP_PER_GEN = 6 # Bigger than 5
SEED = 1000
DEPTH = 2
DECAY = 1

PIECE_VALUES = {
    chess.PAWN: 0,
    chess.KNIGHT: 0,
    chess.BISHOP: 0,
    chess.ROOK: 0,
    chess.QUEEN: 0,
    chess.KING: 0,
}

PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


def blind_mutate_piece_values(SEED):
    """
    Mutates the piece values in a range from 1 to 10000.
    """
    random.seed(SEED)
    
    for piece in PIECES:
        PIECE_VALUES[piece] = random.randint(1, 10000)
        
    return PIECE_VALUES


def guided_mutate_piece_values(new_piece_values, std, SEED):
    random.seed(SEED)
    min_val = 0.0005 * (sum(new_piece_values) / len(new_piece_values))

    for i in range(len(PIECES)):
        piece = PIECES[i]
        center = new_piece_values[i]
        # Multiply by sqrt(3) (~1.732) so uniform sampling in [center - spread, center + spread]
        # preserves the target standard deviation (std * DECAY) instead of shrinking by 42%.
        spread = std[i] * 1.732 * DECAY
        val = center + random.uniform(-1, 1) * spread
        if val < min_val:
            val = min_val

        PIECE_VALUES[piece] = val

    return PIECE_VALUES


def save_piece_values(new_piece_values):
    """Small function to make the learning algorithm use the last learned oiece values if main.py executed
    """
    path = Path("learning_algorithm.py")
    new_block = (
        "PIECE_VALUES = {\n"
        f"    chess.PAWN: {round(float(new_piece_values[0]), 4)},\n"
        f"    chess.KNIGHT: {round(float(new_piece_values[1]), 4)},\n"
        f"    chess.BISHOP: {round(float(new_piece_values[2]), 4)},\n"
        f"    chess.ROOK: {round(float(new_piece_values[3]), 4)},\n"
        f"    chess.QUEEN: {round(float(new_piece_values[4]), 4)},\n"
        "    chess.KING: 0,\n"
        "}"
    )
    text = path.read_text()
    start = text.index("PIECE_VALUES = {")
    end = text.index("}", start) + 1
    path.write_text(text[:start] + new_block + text[end:])


if __name__ == "__main__":
    
    for i in range(1, GENERATIONS + 1):
        
        # Gives 5 times more data to the first completely naive random approach to make it less noisy when doing few experiments per generation
        if i == 1 and EXP_PER_GEN < 25:
            EXP_PER_GEN *= 5
        elif i == 2 and EXP_PER_GEN < 125:
            EXP_PER_GEN //= 5

        for j in range(1, EXP_PER_GEN + 1):
            SEED += 1
            if i == 1:
                mutated_piece_values = blind_mutate_piece_values(SEED)
                run_experiment(mutated_piece_values, DEPTH)
            else:
                mutated_piece_values = guided_mutate_piece_values(new_piece_values, std, SEED)
                run_experiment(mutated_piece_values, DEPTH)

        # If there is an old csv with some data preexsiting it will read the last 1000 rows as well
        if i == 1:
            new_piece_values, mean, std = generation_learning(1000 + EXP_PER_GEN)
        else:
            new_piece_values, mean, std = generation_learning(EXP_PER_GEN)

    save_piece_values(new_piece_values)
    main()
