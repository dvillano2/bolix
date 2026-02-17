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
        if int_shift != 0:
            mask[int_shift] = pointed_wins(
                int_shift, winning_threshold, side, depth
            )
    return mask


############# Opponent masking for removal ####################


def pointed_dir_opponent_mask(
    shift: int,
    length: int,
    winning_threshold: int,
    direction: list[int],
    side: int,
    depth: int,
):
    if length >= winning_threshold:
        raise ValueError("length needs to be less that win threshold")
    interior = valid_board(side, depth) - board_edges(side, depth)
    height, width = interior.shape
    row_shift = shift // width
    col_shift = shift % width
    mask = empty_board(side, depth)
    row_jump = direction[0]
    col_jump = direction[1]
    for i in range(1, length + 1):
        row_spot = int(row_shift + row_jump * i)
        col_spot = int(col_shift + col_jump * i)
        if 0 <= row_spot < height and 0 <= col_spot < width:
            mask[row_spot, col_spot] = 1
        else:
            return empty_board(side, depth)
    return (mask <= interior).all() * mask


def pointed_dir_opponent_mask_all_lengths(
    shift: int,
    winning_threshold: int,
    direction: list[int],
    side: int,
    depth: int,
):
    height = board_height(side, depth)
    width = board_width(side, depth)
    masks = torch.zeros([winning_threshold - 1, height, width])
    for i in range(winning_threshold - 1):
        masks[i] = pointed_dir_opponent_mask(
            shift, i + 1, winning_threshold, direction, side, depth
        )
    return masks


def pointed_opponent_mask(
    shift: int, winning_threshold: int, side: int, depth: int
):
    directions = [[0, 2], [0, -2], [1, 1], [1, -1], [-1, 1], [-1, -1]]
    height = board_height(side, depth)
    width = board_width(side, depth)
    masks = torch.zeros([6, winning_threshold - 1, height, width])
    for i, direction in enumerate(directions):
        masks[i] = pointed_dir_opponent_mask_all_lengths(
            shift, winning_threshold, direction, side, depth
        )
    return masks


def full_opponent_mask(winning_threshold: int, side: int, depth: int):
    height = board_height(side, depth)
    width = board_width(side, depth)
    flattened_shifts = all_shifts(side, depth)
    mask = torch.zeros(
        [height * width, 6, winning_threshold - 1, height, width]
    )
    for shift in flattened_shifts:
        int_shift = int(shift.item())
        if int_shift != 0:
            mask[int_shift] = pointed_opponent_mask(
                int_shift, winning_threshold, side, depth
            )
    return mask


############ Player masks for determining removal ##########


def pointed_dir_player_mask(
    shift: int,
    gap: int,
    winning_threshold: int,
    direction: list[int],
    side: int,
    depth: int,
):
    if gap >= winning_threshold:
        raise ValueError("length needs to be less that win threshold")
    board = valid_board(side, depth)
    height, width = board.shape
    row_shift = shift // width
    col_shift = shift % width
    mask = empty_board(side, depth)
    row_jump = direction[0]
    col_jump = direction[1]
    row_spot = int(row_shift + row_jump * (gap + 1))
    col_spot = int(col_shift + col_jump * (gap + 1))
    if 0 <= row_spot < height and 0 <= col_spot < width:
        mask[row_spot, col_spot] = 1
    else:
        return empty_board(side, depth)
    return (mask <= board).all() * mask


def pointed_dir_player_mask_all_gaps(
    shift: int,
    winning_threshold: int,
    direction: list[int],
    side: int,
    depth: int,
):
    height = board_height(side, depth)
    width = board_width(side, depth)
    masks = torch.zeros([winning_threshold - 1, height, width])
    for i in range(winning_threshold - 1):
        masks[i] = pointed_dir_player_mask(
            shift, i + 1, winning_threshold, direction, side, depth
        )
    return masks


def pointed_player_mask(
    shift: int, winning_threshold: int, side: int, depth: int
):
    directions = [[0, 2], [0, -2], [1, 1], [1, -1], [-1, 1], [-1, -1]]
    height = board_height(side, depth)
    width = board_width(side, depth)
    masks = torch.zeros([6, winning_threshold - 1, height, width])
    for i, direction in enumerate(directions):
        masks[i] = pointed_dir_player_mask_all_gaps(
            shift, winning_threshold, direction, side, depth
        )
    return masks


def full_player_mask(winning_threshold: int, side: int, depth: int):
    height = board_height(side, depth)
    width = board_width(side, depth)
    flattened_shifts = all_shifts(side, depth)
    mask = torch.zeros(
        [height * width, 6, winning_threshold - 1, height, width]
    )
    for shift in flattened_shifts:
        int_shift = int(shift.item())
        if int_shift != 0:
            mask[int_shift] = pointed_player_mask(
                int_shift, winning_threshold, side, depth
            )
    return mask


#### integration of masks with next moves #####


def detect_wins(
    moves: torch.Tensor, player_board: torch.Tensor, wins_mask: torch.Tensor
):
    """make sure the new move is present in the player board"""
    wins_through_move = wins_mask[moves]
    batch, height, width = player_board.shape
    pattern_dims = tuple(range(1, wins_mask.ndim - 2))
    expanded_board = player_board.view(
        batch, *(1,) * len(pattern_dims), height, width
    )
    actual_wins = (wins_through_move <= expanded_board).all(dim=(-2, -1))
    combined_wins = (
        (wins_through_move * actual_wins[..., None, None])
        .sum(dim=pattern_dims)
        .clamp(max=1)
    )
    return combined_wins.any(dim=(-2, -1)), combined_wins


def detect_removal(
    move: int,
    player_board: torch.Tensor,
    opponent_board: torch.Tensor,
    player_removal_mask: torch.Tensor,
    opponent_removal_mask: torch.Tensor,
):
    player_mask_through_move = player_removal_mask[move]
    opponent_mask_through_move = opponent_removal_mask[move]
    batch, height, width = player_board.shape
    pattern_dims = tuple(range(1, player_removal_mask.ndim - 2))
    expanded_player_board = player_board.view(
        batch, *(1,) * len(pattern_dims), height, width
    )
    expanded_opponent_board = opponent_board.view(
        batch, *(1,) * len(pattern_dims), height, width
    )
    actual_removal = (
        (player_mask_through_move <= expanded_player_board)
        & (opponent_mask_through_move <= expanded_opponent_board)
    ).all(dim=(-2, -1))
    possible_removals = (
        opponent_mask_through_move * actual_removal[..., None, None]
    ).sum(dim=pattern_dims)
    return possible_removals.any(dim=(-2, -1)), possible_removals


#### get next arrangement #####


def batch_remove(
    width: int,
    always_invalid: torch.Tensor,
    moves: torch.Tensor,
    state_planes: torch.Tensor,
):
    to_remove = state_planes[:, 1, 0, 0] == 1
    if not to_remove.any():
        return
    removals = state_planes[to_remove]
    removal_moves = moves[to_remove]
    removals[:, 3:, :, :] = torch.roll(removals[:, 3, :, :], shifts=1, dims=1)
    removals[:, 3, :, :] = removals[:, 5, :, :]
    removals[:, 2, :, :] = (
        removals[:, 3, :, :] + removals[:, 4, :, :] + always_invalid
    )
    removals[:, 0, :, :] = 1 - removals[:, 0, :, :]
    removals[:, 1, :, :] = 1 - removals[:, 1, :, :]

    removals[
        torch.arange(len(removals)),
        3,
        removal_moves // width,
        removal_moves % width,
    ] = 0
    state_planes[to_remove] = removals
    return


def apply_moves(
    height: int,
    width: int,
    active: torch.Tensor,
    moves: torch.Tensor,
    state_planes: torch.Tensor,
    wins_mask: torch.Tensor,
    player_removal_mask: torch.Tensor,
    opponent_removal_mask: torch.Tensor,
):
    """
    state_planes is dim: batch, num_planes, height, width
    first plane is all zeros if its white, all one if its black
    the second of these planes is all 0's if it is a placement
    all 1's if its an oppoent removal
    the third is a the forbidden choices, so if its placment
    all the players stones on the board, plus a possible forbidden spot
    from past move removals, and then all invalid spots.
    if its a removal, anything that isn't in the removal mask
    the rest of the planes are player/opponent states alternating.
    the player tensor keeps track of which player is moving for
    state hashing purposes
    """
    pass
