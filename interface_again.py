import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def board_vis(plane_states: torch.Tensor):
    cmap = ListedColormap(["grey", "red", "blue"])
    num_boards, _, height, width = plane_states.shape

    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    fig.suptitle("current game states")

    to_plot = torch.zeros(num_boards, height, width)
    forbidden_mask = plane_states[:, 2, :, :]
    # if the current player is white
    to_plot += (1 - plane_states[:, 0, :, :]) * (
        plane_states[:, 3, :, :] + 2 * plane_states[:, 4, :, :]
    )
    # if the current player is black
    to_plot += plane_states[:, 0, :, :] * (
        2 * plane_states[:, 3, :, :] + plane_states[:, 4, :, :]
    )

    for i in range(num_boards):
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
    plane_states = torch.randint(0, 2, (6, 8, 11, 19))
    plane_states[:, 0, :, :] = torch.zeros(6, 11, 19)
    plane_states[:, 4, :, :] -= plane_states[:, 3, :, :]
    plane_states = torch.clamp(plane_states, min=0)
    plane_states[:, 3, :, :] -= plane_states[:, 4, :, :]
    plane_states = torch.clamp(plane_states, min=0)
    board_vis(plane_states)
