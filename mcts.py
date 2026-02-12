import torch


def get_logits_for_state(state_planes, model):
    """should return a flat tensor of logits as
    well as an overall score"""
    pass


def backtrack(position_eval, path, tree):
    """
    updates the visited nodes
    path should have the form [(index of node, index of move picked)]
    come back to vectorize this
    """
    for node_index, choice in path:
        tree[node_index][1][choice] += 1
        tree[node_index][2][choice] += position_eval
        tree[node_index][3][choice] = (
            tree[node_index][1][choice] / tree[node_index][2][choice]
        )


def pick_next_move(tree, node_index, c_puct):
    """given a tree and a node, give the index of the next move
    as determined by formula below"""
    for_consideration = torch.sqrt(tree[node_index][1]) / (
        1 + tree[node_index][1]
    )
    for_consideration *= c_puct * tree[node_index][0]
    for_consideration += tree[node_index][0]
    return torch.argmax(for_consideration).item()


def intialize(total_nodes, state_planes, model):
    """
    worth noting there are a lot of junk moves, this should be taken care of
    will save on space by a factor of 2
    state planes is the current player, opposite player, stacked some number
    of times topped with a board of ones if the player to move is white,
    zero otherwise
    """
    board_dim = state_planes[0].shape
    total_moves = board_dim[0] * board_dim[1]
    tree = torch.zeros([total_nodes, 4, total_moves])

    hash_storage_size = 1 << (total_nodes.bit_length() + 2)
    hash_storage = torch.zeros(hash_storage_size)
    index_storage = torch.zeros(hash_storage_size)

    return tree, hash_storage, index_storage, hash_storage_size


def add_state(
    state_hash, index, hash_storage, index_storage, hash_storage_size
):
    """
    hash storage size a power of 2
    """
    mask = hash_storage_size - 1
    spot = state_hash & mask
    while hash_storage[spot] != 0:
        spot = (spot + 1) & mask
    hash_storage[spot] = state_hash
    hash_storage[spot] = index


def check_state(state_hash, hash_storage, index_storage, hash_storage_size):
    """
    hash storage size a power of 2
    """
    mask = hash_storage_size - 1
    spot = state_hash & mask
    while hash_storage[spot] != 0 and hash_storage[spot] != state_hash:
        spot = (spot + 1) & mask
    if hash_storage[spot] == 0:
        return -1
    return index_storage[spot]


def pull_current_state(current_planes):
    """
    hash the board to get the state
    """
    pass


def expand_node(
    state_planes,
    model,
    board_hash,
    index,
    tree,
    hash_storage,
    index_storage,
    index_storage_size,
):
    """tree is a big tensor that represents the tree
    creating a new node is putting the logits in
    tree should be initialized to zero
    indexing on a node is as follows
    0: priors P
    1: visit counts N
    2: W accumlated score
    3: Q value
    """
    logits, position_eval = get_logits_for_state(state_planes, model)
    tree[index][0] = logits
    add_state(
        board_hash, index, hash_storage, index_storage, index_storage_size
    )

    return position_eval


def hash_position(move, color, precomputed_hashes):
    pass


def hash_state(current_planes, precomputed_hashes):
    pass


def update_planes(current_planes, move):
    """
    move is an integer indicating where the next move should be...
    need to convert it back to a board position
    """
    dims = current_planes.shape
    num_planes = dims[0]
    board_height = dims[1]
    board_width = dims[2]

    if num_planes > 3:
        indices = torch.arange(1, num_planes, 2)
        current_planes[indices] = torch.roll(current_planes[indices], 1, 0)
        current_planes[1] = current_planes[3]

    spot = (move // board_height, move % board_height)
    current_planes[1][spot[0]][spot[1]] = 1
    pairs = (num_planes - 1) // 2
    current_planes[1::] = (
        current_planes[1::]
        .view(pairs, 2, board_height, board_width)
        .flip(dims=[1])
        .view(num_planes - 1, board_height, board_width)
    )
    current_planes[0] = 1.0 - current_planes[0]


def single_walk(
    tree,
    state_planes,
    starting_hash,
    model,
    hash_storage,
    index_storage,
    hash_storage_size,
    precomputed_hashes,
    c_puct,
    next_index,
):
    """state planes and starting hash should both be"""
    walking_planes = state_planes.clone()
    walking_hash = starting_hash.clone()
    index = check_state(
        walking_hash, hash_storage, index_storage, hash_storage_size
    )
    while index > -1:
        next_move = pick_next_move(tree, index, c_puct)
        next_hash = hash_position(
            next_move, walking_planes[0][0][0], precomputed_hashes
        )
        walking_hash ^= next_hash
        index = check_state(
            walking_hash, hash_storage, index_storage, hash_storage_size
        )
        update_planes(walking_planes, next_move)
    expand_node(
        walking_planes,
        model,
        walking_hash,
        next_index,
        tree,
        hash_storage,
        index_storage,
        hash_storage_size,
    )
    backtrack(....)
