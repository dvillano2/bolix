import torch
from gameplay.utils import Board, all_shifts


def pointed_dir_player_mask(
    shift: int, gap: int, direction: list[int], board: Board
):
    if gap >= board.winning_threshold:
        raise ValueError("length needs to be less that win threshold")
    row_shift, col_shift = shift // board.width, shift % board.width
    row_jump, col_jump = direction
    mask = torch.zeros(board.height, board.width)
    row_spot = int(row_shift + row_jump * (gap + 1))
    col_spot = int(col_shift + col_jump * (gap + 1))
    if 0 <= row_spot < board.height and 0 <= col_spot < board.width:
        mask[row_spot, col_spot] = 1
    else:
        return torch.zeros(board.height, board.width)
    return (mask <= board.valid).all() * mask


def pointed_dir_player_mask_all_gaps(
    shift: int, direction: list[int], board: Board
):
    masks = torch.zeros(
        [board.winning_threshold - 1, board.height, board.width]
    )
    for i in range(board.winning_threshold - 1):
        masks[i] = pointed_dir_player_mask(shift, i + 1, direction, board)
    return masks


def pointed_player_mask(shift: int, board: Board):
    directions = [[0, 2], [0, -2], [1, 1], [1, -1], [-1, 1], [-1, -1]]
    masks = torch.zeros(
        [6, board.winning_threshold - 1, board.height, board.width]
    )
    for i, direction in enumerate(directions):
        masks[i] = pointed_dir_player_mask_all_gaps(shift, direction, board)
    return masks


def full_player_mask(board: Board):
    flattened_shifts = all_shifts(board)
    mask = torch.zeros(
        [
            board.height * board.width,
            6,
            board.winning_threshold - 1,
            board.height,
            board.width,
        ]
    )
    for shift in flattened_shifts:
        int_shift = int(shift.item())
        if int_shift != 0:
            mask[int_shift] = pointed_player_mask(int_shift, board)
    return mask
