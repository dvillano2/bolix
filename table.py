import random

SEED = 23487
rng = random.Random(SEED)


def get_table(height: int, width: int):
    """
    get a number via [a][b][c]
    where a is zero or one (player, opponent)
    b is the row and c is the height
    """
    return [
        [[rng.getrandbits(64) for _ in range(width)] for _ in range(height)]
        for _ in range(2)
    ]
