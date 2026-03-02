import torch
from dataclasses import dataclass


@dataclass
class Board:
    valid: torch.Tensor
    invalid: torch.Tensor
    full: torch.Tensor
    empty: torch.Tensor
    height: int
    width: int
    winning_threshold: int


def board_height(side: int, depth: int) -> int:
    return 2 * depth - 1


def board_width(side: int, depth: int) -> int:
    return (side + depth - 1) * 2 - 1


def empty_board(side: int, depth: int) -> torch.Tensor:
    height = board_height(side, depth)
    width = board_width(side, depth)
    return torch.zeros(height, width)


def valid_board(side: int, depth: int) -> torch.Tensor:
    board = empty_board(side, depth)
    width = board_width(side, depth)
    height = board_height(side, depth)
    current_side = side
    for row in range(height):
        padding = (width - ((2 * (current_side - 1)) + 1)) // 2
        spot = padding
        for _ in range(current_side):
            board[row][spot] = 1
            spot += 2
        if row < depth - 1:
            current_side += 1
        else:
            current_side -= 1
    return board


def make_board(side: int, depth: int, winning_threshold) -> Board:
    valid = valid_board(side, depth)
    invalid = 1 - valid
    height = board_height(side, depth)
    width = board_width(side, depth)
    full = torch.ones(height, width)
    empty = torch.zeros(height, width)
    return Board(valid, invalid, full, empty, height, width, winning_threshold)


def all_shifts(board: Board) -> torch.Tensor:
    """
    note that it impossible for upper left
    (zero indexed when flattened)
    to be valid, so you can just check on nonzero
    to see what is OK
    """
    flattened_board = board.valid.flatten()
    counter = torch.arange(len(flattened_board))
    return counter * flattened_board
