from copy import deepcopy

from .base_agent import Agent, valid_moves
from ..player import other_player
from ..board import CellState


class MinimaxAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
        self._states_visited_last_turn = 0
        self._cached_states = {}

    def next_move(self, board):
        self._states_visited_last_turn = 1 # this counts as a state
        return max(
            [(move,self._minimax_move(move, board, other_player(self._player))) for move in valid_moves(board, self._player)], 
            key=lambda t: t[1] # select min by score in (move, score)
            )[0] # select move in (move, score)

    @property
    def states_visited_last_turn(self):
        return self._states_visited_last_turn

    def _minimax_move(self, move, board, player, depth=0):
        # update board for new "state"
        board.set_cell(move.player, move.row, move.col)
        self._states_visited_last_turn += 1
        score = self._minimax_state(board, player, depth+1)
        board.set_cell(CellState.EMPTY, move.row, move.col)
        return score

    def _minimax_state(self, board, player, depth):
        if self._hash(board) in self._cached_states:
            return self._cached_states[self._hash(board)]

        # terminal cases
        winner = board.winner
        if winner == self._player:
            return 1/depth
        if winner == other_player(self._player):
            return -1/depth
        if not valid_moves(board, player):
            return 0

        # recursive case
        multiplier = 1 if self._player == player else -1 #decide whether to use max or min
        score = multiplier * -2
        for next_move in valid_moves(board, player):
            score = multiplier * max(multiplier * score, multiplier * self._minimax_move(next_move, board, other_player(player),depth))
        self._cached_states[self._hash(board)] = score
        return score
        
    def _hash(self, board):
        value = 0 #ternary hash
        for i in range(board.size ** 2):
            player = board.cell(int(i/board.size), i%board.size)
            if player == self._player:
                value += 2 * (3 ** i)
            elif player == other_player(self._player):
                value += 3 ** i
        return value