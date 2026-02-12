import torch


def initialize(
    batch_size,
    channels,
    board_height,
    board_width,
    total_moves,
    total_nodes,
    model,
):
    priors = torch.zeros(batch_size, total_nodes, total_moves)
    visit_counts = torch.zeros(batch_size, total_nodes, total_moves)
    accumulations = torch.zeros(batch_size, total_nodes, total_moves)

    active = torch.ones(batch_size)
    batch_indexer = torch.arange(batch_size)
    zero_indexer = torch.zeros(batch_size)
    one_indexer = torch.zeros(batch_size)
    indicies = torch.zeros(batch_size)

    hash_storage_size = 1 << (total_nodes.bit_length() + 2)
    hash_storage = torch.zeros(batch_size, hash_storage_size)
    # 2 for storing the scalar model output
    index_storage = torch.zeros(batch_size, hash_storage_size, 2)
    mask = hash_storage_size - 1

    return (
        priors,
        visit_counts,
        accumulations,
        active,
        batch_indexer,
        zero_indexer,
        one_indexer,
        indicies,
        hash_storage,
        index_storage,
        hash_storage_size,
        mask,
    )


def add_board_hashes(
    board_hashes,
    indicies,
    hash_storage,
    index_storage,
    batch_indexer,
    mask,
    zero_indexer,
):
    """
    remember to mask by active searches
    mask is one less than a power of two for fast mod
    might be worth it to come back and deal with recomputing
    values_at_spots each time
    """
    spots = board_hashes & mask
    values_at_spots = hash_storage[batch_indexer, spots]
    not_free = values_at_spots != 0
    while not_free.any():
        spots[not_free] = (spots[not_free] + 1) & mask
        values_at_spots = hash_storage[batch_indexer, spots]
        not_free = values_at_spots != 0
    hash_storage[batch_indexer, spots] = board_hashes
    index_storage[batch_indexer, spots, zero_indexer] = indicies


def check_board_hashes(
    board_hashes,
    indicies,
    hash_storage,
    index_storage,
    batch_indexer,
    mask,
    zero_indexer,
    large_batch,
    hashes_seen,
):
    """
    asdf
    """
    spots = board_hashes & mask
    values_at_spots = hash_storage[batch_indexer, spots]
    other_hash = values_at_spots > 0 and values_at_spots != board_hashes
    while other_hash.any():
        spots[other_hash] = (spots[other_hash] + 1) & mask
        values_at_spots = hash_storage[batch_indexer, spots]
        other_hash = values_at_spots > 0 and values_at_spots != board_hashes

    new_to_batch = values_at_spots == 0
    if large_batch:
        already_present = torch.isin(board_hashes, hash_storage) & new_to_batch
        mover = torch.isin(hash_storage, board_hashes[already_present])


def check_board_hashes(board_hashes, hash_storage, batch_indexer, mask):
    pass


def add_state(board_hashes, indcies, hash_storage, index_storage, mask):
    """
    hash storage size a power of 2
    """
    spots = board_hashes & mask
    while hash_storage[spot] != 0:
        spot = (spot + 1) & mask
    hash_storage[spot] = state_hash
    hash_storage[spot] = index


k
