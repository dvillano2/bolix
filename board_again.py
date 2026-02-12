import torch


def board_height(side: int, depth: int):
    return 2 * depth - 1


def board_width(side: int, depth: int):
    return (side + depth - 1) * 2 - 1


def empty_board(side: int, depth: int):
    height = board_height(side, depth)
    width = board_width(side, depth)
    return torch.zeros([height, width])


def channeled_empty_boards(num_channels: int, side: int, depth: int):
    single_board = empty_board(side, depth)
    return single_board.unsqueeze(0).repeat(num_channels, 1, 1)


def batched_empty_boards(batch_size, num_channels: int, side: int, depth: int):
    channeled_boards = channeled_empty_boards(num_channels, side, depth)
    return channeled_boards.unsqueeze(0).repeat(batch_size, 1, 1, 1)


def valid_board(side: int, depth: int):
    board = empty_board(side, depth)
    width = board_width(side, depth)
    height = board_height(side, depth)
    current_side = side
    for row in range(height):
        padding = (width - ((2 * (current_side - 1)) + 1)) // 2
        spot = padding
        for _ in range(current_side):
            board[row][spot] = 1
            spot += 2
        if row < depth - 1:
            current_side += 1
        else:
            current_side -= 1
    return board


def channeled_valid_boards(num_channels: int, side: int, depth: int):
    single_board = valid_board(side, depth)
    return single_board.unsqueeze(0).repeat(num_channels, 1, 1)


def batched_valid_boards(
    batch_size: int, num_channels: int, side: int, depth: int
):
    channeled_boards = channeled_valid_boards(num_channels, side, depth)
    return channeled_boards.unsqueeze(0).repeat(batch_size, 1, 1, 1)


def channeled_mask(num_channels: int, side: int, depth: int):
    only_valid = channeled_valid_boards(num_channels, side, depth)
    only_valid[0, :, :] = 1
    return only_valid


def batched_mask(batch_size: int, num_channels: int, side: int, depth: int):
    channeled_masks = channeled_mask(num_channels, side, depth)
    return channeled_masks.unsqueeze(0).repeat(batch_size, 1, 1, 1)

def removal_options(own_position, opponent_position, forbidden, move):
    assert own_position[move[0], move[1]] == 0
    assert opponent_position[move[0], move[1]] == 0
    assert forbidden[move[0], move[1]] == 0
