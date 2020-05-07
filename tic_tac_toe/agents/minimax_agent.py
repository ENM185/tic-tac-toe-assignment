from copy import deepcopy

from .base_agent import Agent, valid_moves
from ..player import other_player


class MinimaxAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
        self._states_visited_last_turn = 0

    def next_move(self, board):
        self._states_visited_last_turn = 1 # this counts as a state
        return max(
            [(move,self._minimax_move(move, board, other_player(self._player))) for move in valid_moves(board, self._player)], 
            key=lambda t: t[1] # select min by score in (move, score)
            )[0] # select move in (move, score)

    @property
    def states_visited_last_turn(self):
        return self._states_visited_last_turn

    def _minimax_move(self, move, board, player):
        # update board for new "state"
        board_copy = deepcopy(board).set_cell(move.player, move.row, move.col)
        self._states_visited_last_turn += 1
        return self._minimax_state(board_copy, player)

    def _minimax_state(self, board, player):
        # terminal cases
        winner = board.winner
        if winner == self._player:
            return 1
        if winner == other_player(self._player):
            return -1
        if not valid_moves(board, player):
            return 0

        # recursive case
        multiplier = 1 if self._player == player else -1 #decide whether to use max or min
        score = multiplier * -2
        for next_move in valid_moves(board, player):
            score = multiplier * max(multiplier * score, multiplier * self._minimax_move(next_move, board, other_player(player)))
        return score
        