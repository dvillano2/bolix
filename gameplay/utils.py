import torch


def board_height(side: int, depth: int):
    return 2 * depth - 1


def board_width(side: int, depth: int):
    return (side + depth - 1) * 2 - 1


def empty_board(side: int, depth: int):
    height = board_height(side, depth)
    width = board_width(side, depth)
    return torch.zeros(height, width)


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


def board_data(side: int, depth: int):
    return {
        "height": board_height(side, depth),
        "width": board_width(side, depth),
        "valid_board": valid_board(side, depth),
    }


def pull_board_data(board_data: dict):
    height = board_data["height"]
    width = board_data["width"]
    board = board_data["valid_board"]
    return height, width, board


def board_edges(side: int, depth: int):
    """
    mask that is one on edges of the board,
    zero everywhere else
    """
    board = empty_board(side, depth)
    width = board_width(side, depth)
    height = board_height(side, depth)
    current_side = side
    for row in range(height):
        padding = (width - ((2 * (current_side - 1)) + 1)) // 2
        spot = padding
        for counter in range(current_side):
            if (
                row == 0
                or row == height - 1
                or counter == 0
                or counter == current_side - 1
            ):
                board[row][spot] = 1
            spot += 2
        if row < depth - 1:
            current_side += 1
        else:
            current_side -= 1
    return board


def board_interior_points(side: int, depth: int):
    return valid_board(side, depth) - board_edges(side, depth)
