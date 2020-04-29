from copy import deepcopy
from pprint import pprint

from .base_agent import Agent, valid_moves
from ..player import other_player


class MinimaxAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
        self._states_visited_last_turn = 0

    def next_move(self, board):
        self._states_visited_last_turn = 0
        minimax = self._minimax(deepcopy(board), self._player)
        return minimax[1]

    @property
    def states_visited_last_turn(self):
        return self._states_visited_last_turn

    def _minimax(self, board, player):
        self._states_visited_last_turn += 1

        # base cases (game is over; should never happen on first execution of this method)
        if not valid_moves(board, self._player):
            return (0, None)
        winner = board.winner
        if winner == self._player:
            return (1, None)
        if winner == other_player(self._player):
            return (-1, None)
        
        # create list of tuples: (possible board state, move to get there)
        board_states_with_moves = [
            (deepcopy(board).set_cell(move.player, move.row, move.col), move) 
            for move
            in valid_moves(board, player)]
        # create list of tuples: (score of next move, next move), using opposite of min/max
        # note that we ignore _minimax()[1], as this is 2 moves down
        scores_with_moves = [
            (self._minimax(_board, other_player(player))[0], move)
            for (_board, move) 
            in board_states_with_moves]

        # maximize or minimize move
        if self._player == player:
            return max(scores_with_moves, key=lambda t: t[0])
        return min(scores_with_moves, key=lambda t: t[0])

