from copy import deepcopy
from pprint import pprint

from .base_agent import Agent, valid_moves
from ..player import other_player


class MinimaxAgent(Agent):
    def __init__(self, player, states_visited=0):
        super().__init__(player)
        self._states_visited = states_visited

    def display_stats(self):
        super().display_stats()
        print("\tTotal states visited: {}".format(
            self._states_visited))
        print("\tAverage no. of states visited per move: {}".format(
            self._states_visited / self._moves))

    def next_move(self, board):
        minimax = self._minimax(deepcopy(board), self._player)
        return minimax[1]

    def _minimax(self, board, player):
        self._states_visited += 1

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

