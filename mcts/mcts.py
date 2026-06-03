import torch
from typing import Optional
from gameplay.board import Board
from gameplay.masks import Masks
from gameplay.apply_moves import place_and_remove
from model.model import Model

EXPLORATION_CONSTANT = 1.5


class MCTS:
    def __init__(
        self,
        plane_states: torch.Tensor,
        board: Board,
        masks: Masks,
        total_nodes: int,
        model: Model,  # typing later
    ):
        self.exploration_constant = EXPLORATION_CONSTANT
        self.player: torch.Tensor = plane_states[:, 0, 0, 0]
        self.plane_states: torch.Tensor = plane_states.clone()
        self.local_plane_states: torch.Tensor = plane_states.clone()
        self.batch_size: int = self.plane_states.shape[0]
        self.total_nodes: int = total_nodes
        self.total_moves: int = board.height * board.width
        self.board: Board = board
        self.masks: Masks = masks
        self.model: Model = model

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
        # to save the location of the parent and the move taken
        # to get there
        self.parents = (
            torch.zeros(
                [self.batch_size, total_nodes, 2],
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
        # tracks the sign for backtracking and updating values
        self.sign_tracker = torch.ones(
            [self.batch_size, total_nodes],
            dtype=torch.int32,
        )
        self.sign_indexer = torch.zeros(self.batch_size)
        # FIRST MOVE ANCESTOR, which of possible moves this node
        # came from, used when rolling over the tree
        self.first_move_ancestor = torch.zeros(
            [self.batch_size, total_nodes],
            dtype=torch.int32,
        )
        self.indexer = torch.arange(self.batch_size)
        self.moves = torch.zeros([self.batch_size], dtype=torch.int)

    def establish_root(self):
        logits, _ = self.model.foward(self.plane_states)
        self.policy[self.indexer, self.free_index, :] = logits
        self.free_index += 1

    def call_model(self) -> tuple[torch.Tensor, torch.Tensor]:
        """assumption is that the below funciton will handle wins,
        and make appropirate logits in those places... for the time
        being lets say make them all 2, so if you're walking and hit all 2's
        (eqivalently a single 2)
        and make appropriate values 1
        might not literally be the forward method"""
        logits, values = self.model.foward(self.plane_states)
        return logits, values

    def expand(self, logits: torch.Tensor) -> None:
        not_over = logits[:, 0] != 2
        not_over_indexer = self.indexer[not_over]
        not_over_free_index = self.free_index[not_over]
        self.policy[not_over_indexer, not_over_free_index, :] = logits[
            not_over
        ]
        self.parents[not_over_indexer, not_over_free_index, 0] = (
            self.current_index[not_over]
        )
        self.parents[not_over_indexer, not_over_free_index, 1] = self.moves[
            not_over
        ]
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
        self.children[
            not_over_indexer,
            self.current_index[not_over],
            self.moves[not_over],
        ] = self.free_index[not_over]
        self.current_index[not_over] = self.free_index[not_over]
        self.free_index[not_over] += 1

    def backtrack(self, values: torch.Tensor) -> None:
        # self.parents stores parent and move take from the parent
        parents = self.parents[self.indexer, self.current_index]
        live_parents = parents[:, 0] >= 0
        while (live_parents).any():
            parent_index = parents[:, 0]
            old_moves = parents[:, 1]
            self.visits[self.indexer, parent_index, old_moves] += live_parents
            self.accumulation[self.indexer, parent_index, old_moves] += (
                live_parents
                * values
                * self.sign_tracker[self.indexer, self.sign_indexer]
            )
            self.sign_indexer[live_parents] -= 1
            parents = self.parents[self.indexer, parent_index]
            live_parents = parents[:, 0] >= 0

    def select_moves(self, active_mask: torch.Tensor | None = None):
        if active_mask is not None:
            active_indices = self.indexer[active_mask]
        else:
            active_indices = self.indexer
        w_accumulation = self.accumulation[active_indices, self.current_index]
        n_visits = self.visits[active_indices, self.current_index]
        p_policies = self.policy[active_indices, self.current_index]
        q_values = torch.where(
            n_visits > 0,
            w_accumulation / n_visits,
            torch.zeros_like(w_accumulation),
        )
        n_sum_all_visits = torch.sum(
            n_visits, dim=(-1), keepdim=True
        ).expand_as(n_visits)
        u_exploration_bonus = (
            self.exploration_constant * p_policies
            + torch.sqrt(n_sum_all_visits) / (n_visits + 1)
        )
        self.moves[active_indices] = torch.argmax(
            q_values + u_exploration_bonus, dim=(-1)
        )

    # NEED TO MODIFY TO ACCOUNT FOR FINISHED GAMES
    # Its the two part in the definition of keep walking
    def walk(self) -> torch.Tensor:
        """
        gets self.current index to unexpanded spot or winning spot
        using the MCTS decision function
        """
        # back to considered move
        self.current_index.zero_()
        self.local_plane_states = self.plane_states.clone()

        # set the sign tracker to one and indexer to zeros
        self.sign_tracker.zero_()
        self.sign_tracker += 1
        self.sign_indexer.zero_()

        self.select_moves()
        expanded_and_unexpanded = self.children[
            self.indexer, self.current_index, self.moves
        ]
        keep_walking = (expanded_and_unexpanded) != -1 & (
            expanded_and_unexpanded != 2
        )
        while keep_walking.any():
            walking_indices = self.indexer[keep_walking]
            walking_current_index = self.current_index[keep_walking]
            walking_moves = self.moves[keep_walking]
            self.local_plane_states[keep_walking] = place_and_remove(
                self.local_plane_states[keep_walking],
                self.board,
                self.masks,
                walking_moves,
            )

            other_player = self.local_plane_states[:, 0, 0, 0] != self.player
            self.sign_tracker[keep_walking, other_player] = -1
            self.sign_indexer[keep_walking] += 1

            self.current_index[keep_walking] = self.children[
                walking_indices, walking_current_index, walking_moves
            ]
            self.select_moves(active_mask=keep_walking)
            expanded_and_unexpanded = self.children[
                self.indexer, self.current_index, self.moves
            ]
            keep_walking = (expanded_and_unexpanded) != -1 & (
                expanded_and_unexpanded != 2
            )

        return self.local_plane_states

    def single_simulation(self):
        self.walk()
        logits, values = self.model.forward_with_wins()
        self.expand(logits)
        self.backtrack(values)

    def recommend_moves(self, num_simulations: int | None):
        """
        look into parallelizing (will invovle waiting)
        might not be worth it after profiling, but could be fun
        """
        if num_simulations is None:
            num_simulations = self.total_nodes
        for _ in range(num_simulations):
            self.single_simulation()
        return torch.argmax(self.visits[:, 0], dim=1)
