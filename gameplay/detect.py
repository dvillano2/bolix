import torch
from dataclasses import dataclass


def get_pattern_dims(mask: torch.Tensor):
    return tuple(range(1, mask.ndim - 2))


def get_board_expander(mask: torch.Tensor):
    batch, height, width = mask.shape[0], mask.shape[-2], mask.shape[-1]
    pattern_dims = get_pattern_dims(mask)
    return batch, *(1,) * len(pattern_dims), height, width


@dataclass
class DetectionData:
    pattern_dims: tuple
    board_expander: tuple
    wins_threshold: int


def make_detection_data(mask: torch.Tensor, wins_threshold: int):
    pattern_dims = get_pattern_dims(mask)
    board_expander = get_board_expander(mask)
    return DetectionData(pattern_dims, board_expander, wins_threshold)


def detect_wins(
    moves: torch.Tensor,
    player_board: torch.Tensor,
    wins_mask: torch.Tensor,
    detection_data: DetectionData,
):
    """make sure the new move is present in the player board"""
    wins_through_move = wins_mask[moves]
    expanded_board = player_board.view(detection_data.board_expander)

    actual_wins = (wins_through_move * expanded_board).sum(
        dim=(-2, -1)
    ) == detection_data.wins_threshold

    combined_wins = (
        (wins_through_move * actual_wins[..., None, None])
        .sum(dim=detection_data.pattern_dims)
        .clamp(max=1)
    )
    combined_wins = combined_wins.to(player_board.dtype)
    return combined_wins.any(dim=(-2, -1)), combined_wins


def detect_removal(
    moves: torch.Tensor,
    player_board: torch.Tensor,
    opponent_board: torch.Tensor,
    player_removal_mask: torch.Tensor,
    opponent_removal_mask: torch.Tensor,
    detection_data: DetectionData,
):
    player_mask_through_move = player_removal_mask[moves]
    opponent_mask_through_move = opponent_removal_mask[moves]
    expanded_player_board = player_board.view(detection_data.board_expander)
    expanded_opponent_board = opponent_board.view(
        detection_data.board_expander
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
    ).sum(dim=detection_data.pattern_dims)
    return possible_removals.any(dim=(-2, -1)), possible_removals
