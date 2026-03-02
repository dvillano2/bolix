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
    # interior = valid_board(side, depth) - board_edges(side, depth)
    vb = valid_board(side, depth)
    height, width = vb.shape
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
    # return (mask <= interior).all() * mask
    return (mask <= vb).all() * mask


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
    if (
        0 <= row_spot < height
        and 0 <= col_spot < width
    ):
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
    mask_nonzero = player_mask_through_move.any(dim=(-2, -1))
    actual_removal = (
        (player_mask_through_move <= expanded_player_board)
        & (opponent_mask_through_move <= expanded_opponent_board)
    ).all(dim=(-2, -1)) & mask_nonzero
    possible_removals = (
        opponent_mask_through_move * actual_removal[..., None, None]
    ).sum(dim=pattern_dims)
    return possible_removals.any(dim=(-2, -1)), possible_removals


#### get next arrangement #####


def batch_remove(
    width: int,
    always_invalid: torch.Tensor,
    removals: torch.Tensor,
    removal_moves: torch.Tensor,
) -> torch.Tensor:
    """
    -top plane in player (all zeros or all ones, if top corner is two,
    game is done)
    -second plane is put down a piece (all zeros), or remove (all ones)
    -third plane is forbidden spots (all invalids, all pieces currently
    on the board and a possible extra forbidden from previous removal)
    or winning confguration if the game is over
    -rest is last n spots alternating, so  fourth plane is one on all
    the pieces for the player currently moving, fifth is current opponent
    position etc
    """
    removals[:, 3:, :, :] = torch.roll(removals[:, 3:, :, :], shifts=1, dims=1)
    removals[:, 3, :, :] = removals[:, 5, :, :]

    # switch players
    removals[:, 0, :, :] = 1 - removals[:, 0, :, :]
    # switch to placement
    removals[:, 1, :, :] = 1 - removals[:, 1, :, :]
    # forbidden spots are all current player, all oppent pieces, including the
    # piece to be removed and all the invalid spots
    removals[:, 2, :, :] = (
        removals[:, 3, :, :] + removals[:, 4, :, :] + always_invalid
    )

    # remove the selected piece
    removals[
        torch.arange(len(removals)),
        3,
        removal_moves // width,
        removal_moves % width,
    ] = 0
    return removals


def batch_placement(
    width: int,
    full_board_tensor: torch.tensor,
    always_invalid: torch.Tensor,
    placements: torch.Tensor,
    placement_moves: torch.Tensor,
    wins_mask: torch.Tensor,
    player_removal_mask: torch.Tensor,
    opponent_removal_mask: torch.Tensor,
):
    """
    -top plane in player (all zeros or all ones, if top corner is two,
    game is done)
    -second plane is put down a piece (all zeros), or remove (all ones)
    -third plane is forbidden spots (all invalids, all pieces currently
    on the board and a possible extra forbidden from previous removal)
    or it is winning configuration if the game is over
    -rest is last n spots alternating, so  fourth plane is one on all
    the pieces for the player currently moving, fifth is current opponent
    position etc
    """
    # shift only the player down every other and add the pieces
    placements[:, 3::2, :, :] = torch.roll(
        placements[:, 3::2, :, :], shifts=1, dims=1
    )
    placements[:, 3, :, :] = placements[:, 5, :, :]

    placements[
        torch.arange(len(placements)),
        3,
        placement_moves // width,
        placement_moves % width,
    ] = 1

    # check for wins and mark them if they're there
    wins_present, win_configuration = detect_wins(
        placement_moves, placements[:, 3, :, :], wins_mask
    )

    if wins_present.any():
        placements[wins_present, 1, :, :] = 2
        placements[wins_present, 2, :, :] = win_configuration[wins_present]

    active_mask = ~wins_present
    if not active_mask.any():
        return placements

    active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(1)

    # check for removals
    removal, removal_configuration = detect_removal(
        placement_moves[active_indices],
        placements[active_indices, 3, :, :],
        placements[active_indices, 4, :, :],
        player_removal_mask,
        opponent_removal_mask,
    )

    if removal.any():
        removal_rows = active_indices[removal]
        placements[removal_rows, 1] = 1
        next_forbidden = full_board_tensor - removal_configuration
        placements[removal_rows, 2] = next_forbidden[removal]

    # modify the stack and rest for moves that have no removal
    non_removal_rows = active_indices[~removal]
    placements[non_removal_rows, 0] = 1 - placements[non_removal_rows, 0]
    placements[non_removal_rows, 2] = (
        placements[non_removal_rows, 3]
        + placements[non_removal_rows, 4]
        + always_invalid
    )

    permute = torch.arange(placements.shape[1])
    tail = permute[3:].view(-1, 2)[:, [1, 0]].flatten()
    permute[3:] = tail
    placements[non_removal_rows] = placements[non_removal_rows][:, permute]
    # print(permute)
    # print(not_over.shape)
    # print(not_over[~removal, :, :, :].shape)
    # print(not_over[~removal, permute, :, :].shape)
    # not_over[~removal, :, :, :] = not_over[~removal, permute, :, :]
    return placements


def place_and_remove(
    width: int,
    full_board_tensor: torch.Tensor,
    always_invalid: torch.Tensor,
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
    to_remove = state_planes[:, 1, 0, 0] == 1
    to_place = state_planes[:, 1, 0, 0] == 0

    removals = state_planes[to_remove]
    removal_moves = moves[to_remove]
    removals = batch_remove(width, always_invalid, removals, removal_moves)

    placements = state_planes[to_place]
    placement_moves = moves[to_place]
    placements = batch_placement(
        width,
        full_board_tensor,
        always_invalid,
        placements,
        placement_moves,
        wins_mask,
        player_removal_mask,
        opponent_removal_mask,
    )

    state_planes[to_remove] = removals
    state_planes[to_place] = placements
    return
