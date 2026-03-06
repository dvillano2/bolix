import torch
from gameplay.board import Board
from gameplay.masks import Masks
from model.model import Model


class MCTS:
    def __init__(
        self,
        plane_states: torch.Tensor,
        board: Board,
        masks: Masks,
        total_nodes: int,
        model: Model,  # typing later
    ):
        self.player: torch.Tensor = plane_states[:, 0, 0, 0]
        self.plane_states: torch.Tensor = plane_states.clone()
        self.batch_size: int = self.plane_states.shape[0]
        self.total_nodes: int = total_nodes
        self.total_moves: int = board.height * board.width
        self.model = model

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
        self.first_move_ancestor = torch.zeros(
            [self.batch_size, total_nodes],
            dtype=torch.int32,
        )
        self.indexer = torch.arange(self.batch_size)
        self.moves = torch.zeros([self.batch_size], dtype=torch.int)

    def establish_root(self):
        logits, value = self.model.foward(self.plane_states)
        self.policy[self.indexer, self.free_index, :] = logits
        self.free_index += 1

    def expand(self):
        logits, value = self.model.foward(self.plane_states)
        self.policy[self.indexer, self.free_index, :] = logits
        self.parents[self.indexer, self.free_index] = self.current_index
        # update the first move anscetor
        parent_first_move = self.first_move_ancestor[
            self.indexer, self.current_index
        ]
        real_first_move = torch.where(
            self.current_index == 0, self.moves, parent_first_move
        )
        self.first_move_ancestor[self.indexer, self.free_index] = (
            real_first_move
        )

        ####
        self.children[self.indexer, self.current_index, self.moves] = (
            self.free_index
        )
        self.current_index = self.free_index
        self.free_index += 1
