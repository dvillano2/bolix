import torch
from gameplay.board import Board
from gameplay.masks import Masks
from gameplay.detect import detect_wins, detect_removal


def batch_remove(
    board: Board,
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
        removals[:, 3, :, :] + removals[:, 4, :, :] + board.invalid
    )

    # remove the selected piece
    removals[
        torch.arange(len(removals)),
        3,
        removal_moves // board.width,
        removal_moves % board.width,
    ] = 0
    return removals


def batch_placement(
    board: Board,
    masks: Masks,
    placements: torch.Tensor,
    placement_moves: torch.Tensor,
) -> torch.Tensor:
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
        placement_moves // board.width,
        placement_moves % board.width,
    ] = 1

    # check for wins and mark them if they're there
    wins_present, win_configuration = detect_wins(
        placement_moves, placements[:, 3, :, :], masks, board
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
        masks,
    )

    if removal.any():
        removal_rows = active_indices[removal]
        placements[removal_rows, 1] = 1
        next_forbidden = board.full - removal_configuration
        placements[removal_rows, 2] = next_forbidden[removal]

    # modify the stack and rest for moves that have no removal
    non_removal_rows = active_indices[~removal]
    placements[non_removal_rows, 0] = 1 - placements[non_removal_rows, 0]
    placements[non_removal_rows, 2] = (
        placements[non_removal_rows, 3]
        + placements[non_removal_rows, 4]
        + board.invalid
    )

    permute = torch.arange(placements.shape[1])
    tail = permute[3:].view(-1, 2)[:, [1, 0]].flatten()
    permute[3:] = tail
    placements[non_removal_rows] = placements[non_removal_rows][:, permute]
    return placements


def place_and_remove(
    state_planes: torch.Tensor,
    board: Board,
    masks: Masks,
    moves: torch.Tensor,
) -> torch.Tensor:
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
    removals = batch_remove(board, removals, removal_moves)

    placements = state_planes[to_place]
    placement_moves = moves[to_place]
    placements = batch_placement(
        board,
        masks,
        placements,
        placement_moves,
    )

    state_planes[to_remove] = removals
    state_planes[to_place] = placements
    return state_planes

