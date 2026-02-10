import numpy as np
import torch


def space_per_row(width: int):
    return 4 * width - 3


def display_board(side: int, depth: int):
    middle_width = side + depth - 1
    max_width = space_per_row(middle_width)
    buffer_size = max_width + 6
    current_side = side
    for _ in range(depth):
        sides = (buffer_size - space_per_row(current_side)) // 2
        board_string = (
            " " * sides + "0   " * (current_side - 1) + "0" + " " * sides
        )
        print(board_string)
        current_side += 1
    current_side -= 1
    for _ in range(depth - 1):
        current_side -= 1
        sides = (buffer_size - space_per_row(current_side)) // 2
        board_string = (
            " " * sides + "0   " * (current_side - 1) + "0" + " " * sides
        )
        print(board_string)


def number_board(side: int, depth: int, start: int = 0):
    middle_width = side + depth - 1
    max_width = space_per_row(middle_width)
    buffer_size = max_width + 6
    current_side = side
    counter = start
    for _ in range(depth):
        sides = (buffer_size - space_per_row(current_side)) // 2
        board_string = " " * sides
        for _ in range(current_side - 1):
            board_string += f"{counter}   "
            counter += 1
        board_string += str(counter) + " " * sides
        counter += 1
        print(board_string)
        current_side += 1
    current_side -= 1
    for _ in range(depth - 1):
        current_side -= 1
        sides = (buffer_size - space_per_row(current_side)) // 2
        board_string = " " * sides
        for _ in range(current_side - 1):
            board_string += f"{counter}   "
            counter += 1
        board_string += str(counter) + " " * sides
        counter += 1
        print(board_string)


def board_width(side: int, depth: int):
    return (side + depth - 1) * 2 - 1


def board_height(side: int, depth: int):
    return 2 * depth - 1


def num_board(side: int, depth: int):
    return np.zeros([2 * depth - 1, (side + depth - 1) * 2 - 1], dtype=np.int8)


def valid_spots(side: int, depth: int):
    width = board_width(side, depth)
    height = board_height(side, depth)
    board = np.zeros([height, width])
    count = 1
    current_side = side
    for row in range(height):
        padding = (width - ((2 * (current_side - 1)) + 1)) // 2
        spot = 0
        for _ in range(padding):
            spot += 1
        for _ in range(current_side):
            board[row][spot] = count
            count += 1
            spot += 2
        if row < depth - 1:
            current_side += 1
        else:
            current_side -= 1
    return board


def invalid_mask(side: int, depth: int):
    width = board_width(side, depth)
    height = board_height(side, depth)
    board = torch.zeros([height, width], dtype=torch.bool)
    current_side = side
    for row in range(height):
        padding = (width - ((2 * (current_side - 1)) + 1)) // 2
        spot = 0
        for _ in range(padding):
            spot += 1
        for _ in range(current_side):
            board[row][spot] = True
            spot += 2
        if row < depth - 1:
            current_side += 1
        else:
            current_side -= 1
    return board


def move_invalid_mask(invalid_moves, boards, specials):
    """
    boards is a tensor of size [b, p, h, w], b is batch size,
    p is planes, h is height of board, w is width of board
    want to get out a tensor of size [b, h, w] to apply to the
    logits of the forward of the boards tensor
    atm, special is a tensor of size [b, 2] where 0, 0
    (always a forbidden index unless you board is 1x1....)indicates
    that theres no special forbidden move
    """
    boards_mask = (boards[:, -1, :, :] == 0) & (boards[:, -2, :, :] == 0)
    indexer = torch.arange(len(specials), device=boards.device)
    boards_mask[indexer, specials[:, 0], specials[:, 1]] = False
    mask = invalid_moves & boards_mask
    return mask
