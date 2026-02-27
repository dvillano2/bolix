from gameplay.utils import pull_board_data
import torch


def pointed_dir_win_with_slide(
    shift: int,
    slide: int,
    winning_threshold: int,
    winning_direction: list[int],
    board_data: dict,
):
    """
    winning direction gives the diretion of the win
    so for horizotal its [0, 2]
    one diag is [1, 1]
    the other is [1, -1]
    """
    if slide >= winning_threshold:
        raise ValueError("slide needs to be less that win threshold")
    height, width, board = pull_board_data(board_data)
    row_shift, col_shift = shift // width, shift % width
    row_jump, col_jump = winning_direction
    mask = torch.zeros(height, width)
    for i in range(winning_threshold):
        row_spot = int(row_shift - row_jump * slide + row_jump * i)
        col_spot = int(col_shift - col_jump * slide + col_jump * i)
        if 0 <= row_spot < height and 0 <= col_spot < width:
            mask[row_spot, col_spot] = 1
        else:
            return torch.zeros(height, width)
    return (mask <= board).all() * mask


def pointed_dir_wins(
    shift: int,
    winning_threshold: int,
    winning_direction: list[int],
    board_data: dict,
):
    """
    returns a (winning threhold, height, width) tensor
    which is a mask for all the possible wins going
    though a give point (a shift) in a give direction
    """
    height, width, _ = pull_board_data(board_data)
    masks = torch.zeros(winning_threshold, height, width)
    for i in range(winning_threshold):
        masks[i] = pointed_dir_win_with_slide(
            shift, i, winning_threshold, winning_direction, board_data
        )
    return masks


def pointed_wins(
    shift: int,
    winning_threshold: int,
    board_data: dict,
):
    """
    returns a (3, winning threhold, height, width) tensor
    where the last three dims are describe above and the
    3 is from the 3 different directions
    """
    height, width, _ = pull_board_data(board_data)
    masks = torch.zeros(3, winning_threshold, height, width)
    directions = [[0, 2], [1, 1], [1, -1]]
    for i, winning_direction in enumerate(directions):
        masks[i] = pointed_dir_wins(
            shift, winning_threshold, winning_direction, board_data
        )
    return masks


def all_shifts(board_data: dict):
    """
    note that it impossible for upper left
    (zero indexed when flattened)
    to be valid, so you can just check on nonzero
    to see what is OK
    """
    _, _, board = pull_board_data(board_data)
    flattened_board = board.flatten()
    counter = torch.arange(len(flattened_board))
    return counter * flattened_board


def all_wins(winning_threshold: int, board_data: dict):
    """
    returns a height * width, 3, winning_theshold, height, width
    tensor. If the first entry is zero, means that its not a valid
    board spot and all the rest of the subtensor is also zero.
    if its nonzero, this correpsonds to the integer indexing of a given
    spot and the rest of the subtensor is masks corresponding to winnning
    configurations through that spot
    """
    height, width, _ = pull_board_data(board_data)
    flattened_shifts = all_shifts(board_data)
    mask = torch.zeros([height * width, 3, winning_threshold, height, width])
    for shift in flattened_shifts:
        int_shift = int(shift.item())
        if int_shift != 0:
            mask[int_shift] = pointed_wins(
                int_shift, winning_threshold, board_data
            )
    return mask
