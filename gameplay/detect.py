from dataclasses import dataclass
import torch
from gameplay.masks import Masks, get_expansion_data
from gameplay.board import Board


def detect_wins(
    moves: torch.Tensor, player_board: torch.Tensor, masks: Masks, board: Board
):
    """make sure the new move is present in the player board"""
    wins_through_move = masks.wins_mask[moves]
    pattern_dims, expander = get_expansion_data(wins_through_move)
    expanded_board = player_board.view(expander)

    actual_wins = (wins_through_move * expanded_board).sum(
        dim=(-2, -1)
    ) == board.winning_threshold

    combined_wins = (
        (wins_through_move * actual_wins[..., None, None])
        .sum(dim=pattern_dims)
        .clamp(max=1)
    )
    return combined_wins.any(dim=(-2, -1)), combined_wins


def detect_removal(
    moves: torch.Tensor,
    player_board: torch.Tensor,
    opponent_board: torch.Tensor,
    masks: Masks,
):
    player_mask_through_move = masks.player_mask[moves]
    pattern_dims, player_expander = get_expansion_data(
        player_mask_through_move
    )
    opponent_mask_through_move = masks.opponent_mask[moves]
    _, opponent_expander = get_expansion_data(opponent_mask_through_move)
    expanded_player_board = player_board.view(player_expander)
    expanded_opponent_board = opponent_board.view(opponent_expander)

    # mask_nonzero = player_mask_through_move.any(dim=(-2, -1))
    mask_nonzero = player_mask_through_move.any(dim=(-2, -1))
    actual_removal = (
        (player_mask_through_move <= expanded_player_board)
        & (opponent_mask_through_move <= expanded_opponent_board)
    ).all(dim=(-2, -1)) & mask_nonzero
    # actual_removal = player_condition & opponent_condition & mask_nonzero

    possible_removals = (
        opponent_mask_through_move * actual_removal[..., None, None]
    ).sum(pattern_dims)
    return possible_removals.any(dim=(-2, -1)), possible_removals
