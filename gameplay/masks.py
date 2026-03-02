from dataclasses import dataclass
import torch
from gameplay.board import Board
from gameplay.win_mask import all_wins
from gameplay.opponent_mask import full_opponent_mask
from gameplay.player_mask import full_player_mask


@dataclass
class Masks:
    wins_mask: torch.Tensor
    opponent_mask: torch.Tensor
    player_mask: torch.Tensor
    wins_pattern_dims: tuple
    wins_board_expander: tuple
    position_pattern_dims: tuple
    position_board_expander: tuple


def get_pattern_dims(mask: torch.Tensor):
    return tuple(range(1, mask.ndim - 2))


def get_board_expander(mask: torch.Tensor):
    batch, height, width = mask.shape[0], mask.shape[-2], mask.shape[-1]
    pattern_dims = get_pattern_dims(mask)
    return batch, *(1,) * len(pattern_dims), height, width


def make_masks(board: Board):
    wins_mask = all_wins(board)
    opponent_mask = full_opponent_mask(board)
    player_mask = full_player_mask(board)
    wins_pattern_dims = get_pattern_dims(wins_mask)
    wins_board_expander = get_board_expander(wins_mask)
    position_pattern_dims = get_pattern_dims(player_mask)
    position_board_expander = get_board_expander(player_mask)
    return Masks(
        wins_mask,
        opponent_mask,
        player_mask,
        wins_pattern_dims,
        wins_board_expander,
        position_pattern_dims,
        position_board_expander,
    )
