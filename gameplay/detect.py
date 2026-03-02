import torch
from dataclasses import dataclass
from gameplay.masks import Masks
from gameplay.board import Board


def detect_wins(
    moves: torch.Tensor, player_board: torch.Tensor, masks: Masks, board: Board
):
    """make sure the new move is present in the player board"""
    wins_through_move = masks.wins_mask[moves]
    expanded_board = player_board.view(masks.board_expander)

    actual_wins = (wins_through_move * expanded_board).sum(
        dim=(-2, -1)
    ) == Board.winning_threshold

    combined_wins = (
        (wins_through_move * actual_wins[..., None, None])
        .sum(dim=masks.wins_pattern_dims)
        .clamp(max=1)
    )
    combined_wins = combined_wins.to(player_board.dtype)
    return combined_wins.any(dim=(-2, -1)), combined_wins


def detect_removal(
    moves: torch.Tensor,
    player_board: torch.Tensor,
    opponent_board: torch.Tensor,
    masks: Masks,
):
    player_mask_through_move = masks.player_mask[moves]
    opponent_mask_through_move = masks.opponent_mask[moves]
    expanded_player_board = player_board.view(masks.position_board_expander)
    expanded_opponent_board = opponent_board.view(
        masks.position_board_expander
    )

    # mask_nonzero = player_mask_through_move.any(dim=(-2, -1))
    player_condition = (player_mask_through_move * expanded_player_board).sum(
        dim=(-2, -1)
    ) == 2
    opponent_condition = (
        (opponent_mask_through_move * expanded_opponent_board)
        .eq(opponent_mask_through_move)
        .all(dim=(-2, -1))
    )
    actual_removal = player_condition & opponent_condition
    # actual_removal = player_condition & opponent_condition & mask_nonzero

    possible_removals = (
        opponent_mask_through_move * actual_removal[..., None, None]
    ).sum(dim=masks.position_pattern_dims)
    return possible_removals.any(dim=(-2, -1)), possible_removals
