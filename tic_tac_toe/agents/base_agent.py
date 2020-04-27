from abc import ABC, abstractmethod
from collections import namedtuple
import time

from ..player import PLAYER_NAMES


class Move(namedtuple("Move", ["player", "row", "col"])):
    def __repr__(self):
        return "Move(player={},row={},col={})".format(
            PLAYER_NAMES[self.player], self.row, self.col)


class Agent(ABC):
    def __init__(self, player, runtime=0, moves=0):
        self._player = player
        self._runtime = runtime
        self._moves = moves

    @property
    def name(self):
        return PLAYER_NAMES[self._player]

    @property
    def average_runtime(self):
        assert self._moves != 0
        return self._runtime / self._moves

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
