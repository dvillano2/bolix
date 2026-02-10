def neighbor_directions():
    return [[1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]


def inside(y, x, height, width):
    return 0 <= y < height and 0 <= x < width


def check_next(point, direction, height, width):
    y, x = point
    d0, d1 = direction
    return inside(y + d0, x + d1, height, width)


def advance(point, direction):
    y, x = point
    d0, d1 = direction
    return [y + d0, x + d1]


def check_advance_plane(point, height, width, direction, plane):
    if not check_next(point, direction, height, width):
        return False
    y, x = advance(point, direction)
    if plane[y][x] != 1:
        return False
    return True


def check_direction(
    point, height, width, direction, own_plane, opponent_plane
):
    current_point = point
    possible_spots = []
    while check_advance_plane(
        current_point, height, width, direction, opponent_plane
    ):
        current_point = advance(point, direction)
        possible_spots.append(point)
    if check_advance_plane(current_point, height, width, direction, own_plane):
        return possible_spots
    return []


def check_removal(point, height, width, directions, own_plane, opponent_plane):
    spots = []
    for direction in directions:
        spots.extend(
            check_direction(
                point, height, width, direction, own_plane, opponent_plane
            )
        )
    return spots


def check_game_over(point, height, width, directions, own_plane, to_win=5):
    for direction in directions:
        current_point = point
        count = 1
        if check_advance_plane(
            current_point, height, width, direction, own_plane
        ):
            count += 1
            if count == to_win:
                return True
            current_point = advance(current_point, direction)
    return False
