from abc import ABC, abstractmethod
from collections import namedtuple
import time

from ..player import PLAYER_NAMES


class Move(namedtuple("Move", ["player", "row", "col"])):
    def __repr__(self):
        return "Move(player={},row={},col={})".format(
            PLAYER_NAMES[self.player], self.row, self.col)


class Agent(ABC):
    def __init__(self, player, runtime=0, moves=0, wins=0, games=0):
        self._player = player
        self._runtime = runtime
        self._moves = moves
        self._wins = wins
        self._games = games

    def display_stats(self):
        print("Player {} stats:".format(PLAYER_NAMES[self._player]))
        print("\t{} of {} games won".format(self._wins, self._games))
        print("\tTotal Moves: {}".format(self._moves))
        print("\tTotal Runtime: {} seconds".format(self._runtime))
        print("\tAverage Runtime: {} seconds".format(self._runtime/self._moves))

    def finish_game(self):
        self._games += 1

    def add_win(self):
        self._wins += 1

    def next_move_timed(self, board):
        start = time.time()
        move = self.next_move(board)
        end = time.time()

        self._moves += 1
        self._runtime += end - start

        return move

    @abstractmethod
    def next_move(self, board):
        pass


def valid_moves(board, player):
    return [Move(player, i, j)
            for i, j in board.empty_cells]
