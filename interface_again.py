from datetime import datetime
from pathlib import Path
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from board_again import (
    board_height,
    board_width,
    all_wins,
    full_opponent_mask,
    full_player_mask,
    valid_board,
    place_and_remove,
)

ROWS = 2
COLS = 3


class BoardVisualizer:
    def __init__(self, save: bool = False, **config):
        self.side = config["side"]
        self.depth = config["depth"]
        self.height = board_height(self.side, self.depth)
        self.width = board_width(self.side, self.depth)
        self.winning_threshold = config["winning_threshold"]
        self.valid_board = valid_board(self.side, self.depth)
        self.always_invalid = 1 - self.valid_board
        self.full_board = torch.ones(self.height, self.width)

        self.boards = config["boards"]
        self.plane_depth = config["plane_depth"]

        self.plane_states = torch.zeros(
            self.boards, self.plane_depth, self.height, self.width
        )
        self.plane_states[:, 2, :, :] = self.always_invalid
        self.moves = torch.zeros(self.boards, dtype=int) - 1

        self.all_wins_mask = all_wins(
            self.winning_threshold, self.side, self.depth
        )
        self.opponent_mask = full_opponent_mask(
            self.winning_threshold, self.side, self.depth
        )
        self.player_mask = full_player_mask(
            self.winning_threshold, self.side, self.depth
        )

        self.count = 0
        self.history: dict = {}
        self.save = save
        if self.save:
            self.history["side"] = self.side
            self.history["depth"] = self.depth
            self.history["winning_threshold"] = self.winning_threshold
            self.history["boards"] = self.boards
            self.history[self.count] = {}
            self.history[self.count]["state"] = self.plane_states

        # plotting stuff
        self.cmap = ListedColormap(
            ["grey", "red", "blue", "coral", "deeppink", "black"]
        )
        self.norm = BoundaryNorm(
            boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            ncolors=self.cmap.N,
        )
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 6))
        self.fig.suptitle("current game states")
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.to_plot = torch.zeros(self.boards, self.height, self.width)
        self.base_images = [[None for _ in range(COLS)] for _ in range(ROWS)]

        self.overlay = [None for _ in range(self.boards)]
        self.overlay_images = [
            [None for _ in range(COLS)] for _ in range(ROWS)
        ]

        self.next_moves = np.full(
            (self.boards, self.height, self.width), np.nan, dtype=float
        )
        self.next_move_images = [
            [None for _ in range(COLS)] for _ in range(ROWS)
        ]

        for r in range(ROWS):
            for c in range(COLS):
                index = r * COLS + c
                board = self.to_plot[index]
                self.base_images[r][c] = self.axes[r, c].imshow(
                    board.numpy(),
                    cmap=self.cmap,
                    norm=self.norm,
                )
                self.overlay_images[r][c] = self.axes[r, c].imshow(
                    board.numpy(),
                    cmap=self.cmap,
                    norm=self.norm,
                )
                masked = np.ma.masked_invalid(self.next_moves[index])
                self.next_move_images[r][c] = self.axes[r, c].imshow(
                    masked,
                    cmap=self.cmap,
                    norm=self.norm,
                    zorder=10,
                )

        self.update_fig()

        plt.tight_layout()
        plt.show()

    def _update_pieces(self):
        self.to_plot = torch.zeros(self.boards, self.height, self.width)
        # if the current player is white
        self.to_plot += (1 - self.plane_states[:, 0, :, :]) * (
            self.plane_states[:, 3, :, :] + 2 * self.plane_states[:, 4, :, :]
        )
        # if the current player is black
        self.to_plot += self.plane_states[:, 0, :, :] * (
            2 * self.plane_states[:, 3, :, :] + self.plane_states[:, 4, :, :]
        )

    def _update_unoccupied_forbidden(self):
        unoccupied_forbidden = torch.clamp(
            self.plane_states[:, 2, :, :] - self.to_plot, min=0
        )
        for i in range(self.boards):
            self.overlay[i] = np.where(
                unoccupied_forbidden[i].numpy() == 1, 5.5, np.nan
            )

    def update_fig(self):
        self._update_pieces()
        self._update_unoccupied_forbidden()
        for i in range(self.boards):
            r = i // COLS
            c = i % COLS
            board = self.to_plot[i]
            self.base_images[r][c].set_data(board)
            overlay = self.overlay[i]
            self.overlay_images[r][c].set_data(overlay)
            masked = np.ma.masked_invalid(self.next_moves[i])
            self.next_move_images[r][c].set_data(masked)
            self.next_move_images[r][c].set_visible(False)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes is None:
            return
        for r in range(ROWS):
            for c in range(COLS):
                self.next_move_images[r][c].set_visible(True)
                board = r * COLS + c
                if event.inaxes == self.axes[r, c]:
                    base_image = self.base_images[r][c]
                    row_index, col_index = self.click_to_square(
                        event, base_image
                    )
                    if self.plane_states[board, 2, row_index, col_index] == 1:
                        # some kind of warning, this is not allowed
                        pass
                    else:
                        spot = row_index * self.width + col_index
                        self.moves[board] = spot
                        self.render_next_moves(board)
        return

    def click_to_square(self, event, image):
        x_min, x_max, y_min, y_max = image.get_extent()
        col_index = int((event.xdata - x_min) / (x_max - x_min) * self.width)
        row_index = int((event.ydata - y_min) / (y_max - y_min) * self.height)
        return self.height - 1 - row_index, col_index

    def render_next_moves(self, board):
        self._update_next_moves(board)

        r = board // COLS
        c = board % COLS

        masked = np.ma.masked_invalid(self.next_moves[board])
        self.next_move_images[r][c].set_data(masked)

        self.fig.canvas.draw_idle()

    def _update_next_moves(self, board):
        self.next_moves[board, :, :] = np.nan
        move = self.moves[board]
        if move.item() < 0:
            return
        move_row = int(move // self.width)
        move_col = int(move % self.width)
        self.next_moves[board, move_row, move_col] = 4

    def apply_moves(self):
        place_and_remove(
            self.width,
            self.full_board,
            self.always_invalid,
            self.moves,
            self.plane_states,
            self.all_wins_mask,
            self.player_mask,
            self.opponent_mask,
        )
        if self.save:
            self.history[self.count]["moves"] = self.moves
            self.count += 1
            self.history[self.count] = {}
            self.history[self.count]["state"] = self.plane_states
            if self.count == 25:
                now = datetime.now()
                formatted_string = now.strftime("%Y_%m_%d%H%M%S")
                file_name = formatted_string + ".pkl"
                project_home = Path.cwd()
                file = project_home / "tests" / "examples" / file_name
                with open(file, "wb") as f:
                    pickle.dump(self.history, f)

        self.next_moves = np.full(
            (self.boards, self.height, self.width), np.nan, dtype=float
        )
        self.moves = torch.zeros(self.boards, dtype=int) - 1
        self.update_fig()

    def on_key(self, event):
        if event.key == "enter":
            self.apply_moves()


if __name__ == "__main__":
    torch.set_printoptions(threshold=torch.inf)
    board_config = {
        "boards": 6,
        "side": 5,
        "depth": 6,
        "winning_threshold": 5,
        "plane_depth": 13,
    }
    BoardVisualizer(False, **board_config)
