import torch
import torch.nn.functional as F


def board_height(side: int, depth: int):
    return 2 * depth - 1


def board_width(side: int, depth: int):
    return (side + depth - 1) * 2 - 1


def empty_board(side: int, depth: int):
    height = board_height(side, depth)
    width = board_width(side, depth)
    return torch.zeros([height, width])


def channeled_empty_boards(num_channels: int, side: int, depth: int):
    single_board = empty_board(side, depth)
    return single_board.unsqueeze(0).repeat(num_channels, 1, 1)


def batched_empty_boards(batch_size, num_channels: int, side: int, depth: int):
    channeled_boards = channeled_empty_boards(num_channels, side, depth)
    return channeled_boards.unsqueeze(0).repeat(batch_size, 1, 1, 1)


def valid_board(side: int, depth: int):
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


def board_edges(side: int, depth: int):
    """
    mask that is one on edges of the board,
    zero everywhere else
    """
    board = empty_board(side, depth)
    width = board_width(side, depth)
    height = board_height(side, depth)
    current_side = side
    for row in range(height):
        padding = (width - ((2 * (current_side - 1)) + 1)) // 2
        spot = padding
        for counter in range(current_side):
            if (
                row == 0
                or row == height - 1
                or counter == 0
                or counter == current_side - 1
            ):
                board[row][spot] = 1
            spot += 2
        if row < depth - 1:
            current_side += 1
        else:
            current_side -= 1
    return board


def board_interior_points(side: int, depth: int):
    return valid_board(side, depth) - board_edges(side, depth)


############# MASKS FOR WINNING PART ################################


def shift_board(board, shift):
    """
    shift is given as integer
    """
    height, _ = board.shape
    row_shift = shift // height
    col_shift = shift % height
    return F.pad(board, (row_shift, 0, col_shift, 0))


def pointed_dir_win_with_slide(
    shift: int,
    slide: int,
    winning_threshold: int,
    winning_direction: list[int],
    side: int,
    depth: int,
):
    """
    winning direction gives the diretion of the win
    so for horizotal its [0, 2]
    one diag is [1, 1]
    the other is [1, -1]
    """
    if slide >= winning_threshold:
        raise ValueError("slide needs to be less that win threshold")
    board = valid_board(side, depth)
    height, width = board.shape
    row_shift = shift // width
    col_shift = shift % width
    mask = empty_board(side, depth)
    row_jump = winning_direction[0]
    col_jump = winning_direction[1]
    for i in range(winning_threshold):
        row_spot = int(row_shift - row_jump * slide + row_jump * i)
        col_spot = int(col_shift - col_jump * slide + col_jump * i)
        if 0 <= row_spot < height and 0 <= col_spot < width:
            mask[row_spot, col_spot] = 1
        else:
            return empty_board(side, depth)
    return (mask <= board).all() * mask


def pointed_dir_wins(
    shift: int,
    winning_threshold: int,
    winning_direction: list[int],
    side: int,
    depth: int,
):
    """
    returns a (winning threhold, height, width) tensor
    which is a mask for all the possible wins going
    though a give point (a shift) in a give direction
    """
    masks = empty_board(side, depth)[None, ...].repeat(winning_threshold, 1, 1)
    for i in range(winning_threshold):
        masks[i, :, :] = pointed_dir_win_with_slide(
            shift, i, winning_threshold, winning_direction, side, depth
        )
    return masks


def pointed_wins(
    shift: int,
    winning_threshold: int,
    side: int,
    depth: int,
):
    """
    returns a (3, winning threhold, height, width) tensor
    where the last three dims are describe above and the
    3 is from the 3 different directions
    """
    masks = empty_board(side, depth)[None, None, ...].repeat(
        3, winning_threshold, 1, 1
    )
    directions = [[0, 2], [1, 1], [1, -1]]
    for i, winning_direction in enumerate(directions):
        masks[i, :, :, :] = pointed_dir_wins(
            shift, winning_threshold, winning_direction, side, depth
        )
    return masks


def all_shifts(side: int, depth: int):
    """
    note that it impossible for upper left
    (zero indexed when flattened)
    to be valid, so you can just check on nonzero
    to see what is OK
    """
    board = valid_board(side, depth)
    flattened_board = board.flatten()
    counter = torch.arange(len(flattened_board))
    return counter * flattened_board


def all_wins(winning_threshold: int, side: int, depth: int):
    """
    returns a height * width, 3, winning_theshold, height, width
    tensor. If the first entry is zero, means that its not a valid
    board spot and all the rest of the subtensor is also zero.
    if its nonzero, this correpsonds to the integer indexing of a given
    spot and the rest of the subtensor is masks corresponding to winnning
    configurations through that spot
    """
    height = board_height(side, depth)
    width = board_width(side, depth)
    flattened_shifts = all_shifts(side, depth)
    mask = torch.zeros([height * width, 3, winning_threshold, height, width])
    for shift in flattened_shifts:
        int_shift = int(shift.item())
        if int_shift != 1:
            mask[int_shift] = pointed_wins(
                int_shift, winning_threshold, side, depth
            )
    return mask
