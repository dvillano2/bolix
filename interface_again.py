import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from board_again import (
    board_height,
    board_width,
    all_wins,
    full_opponent_mask,
    full_player_mask,
    valid_board,
)


class BoardVisualizer:
    def __init__(self, **config):
        self.side = config["side"]
        self.depth = config["depth"]
        self.height = board_height(self.side, self.depth)
        self.width = board_width(self.side, self.depth)
        self.winning_threshold = config["winning_threshold"]

        self.boards = config["boards"]
        self.plane_depth = config["plane_depth"]

        self.plane_states = torch.zeros(
            self.boards, self.plane_depth, self.height, self.width
        )
        self.plane_states[:, 2:, :, :] = 1 - valid_board(self.side, self.depth)
        self.moves = torch.zeros(self.boards)

        self.all_wins_mask = all_wins(
            self.winning_threshold, self.side, self.depth
        )
        self.opponent_mask = full_opponent_mask(
            self.winning_threshold, self.side, self.depth
        )
        self.player_mask = full_player_mask(
            self.winning_threshold, self.side, self.depth
        )
        self.draw_board()

    def draw_board(self):
        cmap = ListedColormap(["grey", "red", "blue"])

        fig, axes = plt.subplots(2, 3, figsize=(15, 6))
        fig.suptitle("current game states")

        to_plot = torch.zeros(self.boards, self.height, self.width)
        forbidden_mask = self.plane_states[:, 2, :, :]
        # if the current player is white
        to_plot += (1 - self.plane_states[:, 0, :, :]) * (
            self.plane_states[:, 3, :, :] + 2 * self.plane_states[:, 4, :, :]
        )
        # if the current player is black
        to_plot += self.plane_states[:, 0, :, :] * (
            2 * self.plane_states[:, 3, :, :] + self.plane_states[:, 4, :, :]
        )

        for i in range(self.boards):
            r = i // 3
            c = i % 3
            board = to_plot[i]
            forbidden = torch.clamp(forbidden_mask[i] - board, min=0)
            overlay = np.where(forbidden.numpy() == 1, 1.0, np.nan)
            print(forbidden)

            axes[r, c].imshow(
                board.numpy(),
                cmap=cmap,
            )
            axes[r, c].imshow(overlay, cmap="grey", alpha=0.9)
            axes[r, c].set_title(f"Game {i+1}")
            axes[r, c].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    board_config = {
        "boards": 6,
        "side": 5,
        "depth": 6,
        "winning_threshold": 5,
        "plane_depth": 13,
    }
    BoardVisualizer(**board_config)
