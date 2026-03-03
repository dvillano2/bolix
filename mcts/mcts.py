import torch
from gameplay.board import Board
from gameplay.masks import Masks


class MCTS:
    def __init__(
        self,
        plane_states: torch.Tensor,
        board: Board,
        masks: Masks,
        total_nodes: int,
    ):
        self.player: torch.Tensor = plane_states[:, 0, 0, 0]
        self.current_states: torch.Tensor = plane_states.clone()
        self.batch_size: int = self.current_states.shape[0]
        self.total_nodes: int = total_nodes
        self.total_moves: int = board.height * board.width

        # POLICIES P in lit
        self.policy = torch.zeros(
            [self.batch_size, total_nodes, self.total_moves],
            dtype=torch.float32,
        )
        # VISITS, N in lit
        self.visits = torch.zeros(
            [self.batch_size, total_nodes, self.total_moves],
            dtype=torch.float32,
        )
        # ACCUMULATION, W in lit
        self.accumulation = torch.zeros(
            [self.batch_size, total_nodes, self.total_moves],
            dtype=torch.float32,
        )
        # CHILDREN, set to -1 if none yet
        self.children = (
            torch.zeros(
                [self.batch_size, total_nodes, self.total_moves],
                dtype=torch.int32,
            )
            - 1
        )
        # PARENTS, set to -1 if none yet
        self.parents = (
            torch.zeros(
                [self.batch_size, total_nodes],
                dtype=torch.int32,
            )
            - 1
        )
        # CURRENT INDEX, current position on each walk
        self.current_index = torch.zeros(
            [self.batch_size],
            dtype=torch.int32,
        )
        # NEXT FREE INDEX, where to put the next expanded node
        self.free_index = torch.zeros(
            [self.batch_size],
            dtype=torch.int32,
        )
        # FIRST MOVE ANCESTOR, which of possible moves this node
        # came from, used when rolling over the tree
        self.parents = torch.zeros(
            [self.batch_size, total_nodes],
            dtype=torch.int32,
        )
