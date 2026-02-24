import torch
import matplotlib.pyplot as plt
from board_again import board_height, board_width, empty_board, valid_board

HEIGHT = board_height(5, 6)
WIDTH = board_width(5, 6)
VALID_BOARD = valid_board(5, 6)


def draw_board(ax, identity_plane, player_plane, opponent_plane):
    """
    player_plane, opponent_plane: H x W torch tensors (0 or 1)
    """
    board = torch.zeros(HEIGHT, WIDTH, 3)  # RGB float tensor

    if identity_plane[0, 0] == 0:
        board[player_plane > 0] = torch.tensor([0.0, 0.0, 1.0])
        board[opponent_plane > 0] = torch.tensor([1.0, 0.0, 0.0])
    else:
        board[player_plane > 0] = torch.tensor([1.0, 0.0, 0.0])
        board[opponent_plane > 0] = torch.tensor([0.0, 0.0, 1.0])

    ax.imshow(board.numpy(), origin='upper', aspect='auto', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])


def draw_forbidden(ax, forbidden_plane):
    """
    Overlay forbidden spots as semi-transparent X.
    forbidden_plane: H x W torch tensor (0 or >0)
    """
    ys, xs = torch.nonzero(forbidden_plane > 0, as_tuple=True)
    ax.scatter(
        xs.numpy(), ys.numpy(), marker="x", color="gray", s=100, alpha=0.5
    )


def draw_game(ax, state_planes, game_idx):
    """
    state_planes: B x C x H x W torch tensor
    game_idx: which game in batch
    """
    draw_board(
        ax,
        state_planes[game_idx, 0],
        state_planes[game_idx, 3],
        state_planes[game_idx, 4],
    )
    draw_forbidden(ax, state_planes[game_idx, 2])


# Example
B, C, H, W = 10, 13, 11, 19
planes = torch.zeros(B, C, H, W, dtype=torch.int)
planes[:, 2, :, :] = 1 - VALID_BOARD

fig, axes = plt.subplots(2, 5, figsize=(24, 10), constrained_layout=False)
axes = axes.flatten()

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")

fig.patch.set_facecolor("white")

plt.subplots_adjust(left=0.01, right=0.99,
                    top=0.99, bottom=0.01,
                    wspace=0.02, hspace=0.02)

for i, ax in enumerate(axes):
    draw_game(ax, planes, i)

# Remove all spacing
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.2, hspace=0.5)

plt.show()
