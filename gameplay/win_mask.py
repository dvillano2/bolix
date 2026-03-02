from gameplay.board import Board, all_shifts
import torch


def pointed_dir_win_with_slide(
    shift: int,
    slide: int,
    direction: list[int],
    board: Board,
):
    """
    winning direction gives the diretion of the win
    so for horizotal its [0, 2]
    one diag is [1, 1]
    the other is [1, -1]
    """
    if slide >= board.winning_threshold:
        raise ValueError("slide needs to be less that win threshold")
    row_shift, col_shift = divmod(shift, board.width)
    row_jump, col_jump = direction
    mask = torch.zeros(board.height, board.width)
    for i in range(board.winning_threshold):
        row_spot = int(row_shift - row_jump * slide + row_jump * i)
        col_spot = int(col_shift - col_jump * slide + col_jump * i)
        if 0 <= row_spot < board.height and 0 <= col_spot < board.width:
            mask[row_spot, col_spot] = 1
        else:
            return board.empty
    return (mask <= board.valid).all() * mask


def pointed_dir_wins(
    shift: int,
    direction: list[int],
    board: Board,
):
    """
    returns a (winning threhold, height, width) tensor
    which is a mask for all the possible wins going
    though a give point (a shift) in a give direction
    """
    masks = torch.zeros(board.winning_threshold, board.height, board.width)
    for i in range(board.winning_threshold):
        masks[i] = pointed_dir_win_with_slide(shift, i, direction, board)
    return masks


def pointed_wins(shift: int, board: Board):
    """
    returns a (3, winning threhold, height, width) tensor
    where the last three dims are describe above and the
    3 is from the 3 different directions
    """
    masks = torch.zeros(3, board.winning_threshold, board.height, board.width)
    directions = [[0, 2], [1, 1], [1, -1]]
    for i, direction in enumerate(directions):
        masks[i] = pointed_dir_wins(shift, direction, board)
    return masks


def all_wins(board: Board):
    """
    returns a height * width, 3, winning_theshold, height, width
    tensor. If the first entry is zero, means that its not a valid
    board spot and all the rest of the subtensor is also zero.
    if its nonzero, this correpsonds to the integer indexing of a given
    spot and the rest of the subtensor is masks corresponding to winnning
    configurations through that spot
    """
    flattened_shifts = all_shifts(board)
    mask = torch.zeros(
        [
            board.height * board.width,
            3,
            board.winning_threshold,
            board.height,
            board.width,
        ]
    )
    for shift in flattened_shifts:
        int_shift = int(shift.item())
        if int_shift != 0:
            mask[int_shift] = pointed_wins(int_shift, board)
    return mask
