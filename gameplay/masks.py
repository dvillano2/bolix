from dataclasses import dataclass
import torch
from gameplay.board import Board
from gameplay.specific_masks.win_mask import all_wins
from gameplay.specific_masks.opponent_mask import full_opponent_mask
from gameplay.specific_masks.player_mask import full_player_mask


@dataclass
class Masks:
    wins_mask: torch.Tensor
    opponent_mask: torch.Tensor
    player_mask: torch.Tensor


def get_pattern_dims(mask: torch.Tensor) -> tuple[int, ...]:
    return tuple(range(1, mask.ndim - 2))


def get_expansion_data(
    mask: torch.Tensor,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    batch_size, height, width = mask.shape[0], mask.shape[-2], mask.shape[-1]
    pattern_dims = get_pattern_dims(mask)
    return pattern_dims, (batch_size, *(1,) * len(pattern_dims), height, width)


def make_masks(board: Board) -> Masks:
    wins_mask = all_wins(board)
    opponent_mask = full_opponent_mask(board)
    player_mask = full_player_mask(board)
    return Masks(
        wins_mask,
        opponent_mask,
        player_mask,
    )
