import torch
import matplotlib.pyplot as plt


def board_vis(plane_states: torch.Tensor):
    num_boards, _, height, width = plane_states.shape

    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    fig.suptitle("current game states")

    to_plot = torch.zeros(num_boards, height, width)
    to_plot += 10 * plane_states[:, 2, :, :]
    # if the current player is white
    to_plot += (1 - plane_states[:, 0, :, :]) * (
        10 * plane_states[:, 3, :, :] + 50 * plane_states[:, 4, :, :]
    )
    # if the current player is black
    to_plot += plane_states[:, 0, :, :] * (
        50 * plane_states[:, 3, :, :] + 10 * plane_states[:, 4, :, :]
    )

    for i in range(num_boards):
        r = i // 3
        c = i % 3
        board = to_plot[i]
        im = axes[r, c].imshow(
            board.numpy(),
            cmap="viridis",
        )
        axes[r, c].set_title(f"Game {i+1}")
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plane_states = torch.randn(6, 8, 11, 19)
    board_vis(plane_states)
