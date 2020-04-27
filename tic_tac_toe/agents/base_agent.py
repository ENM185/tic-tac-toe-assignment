from abc import ABC, abstractmethod
from collections import namedtuple

from ..player import PLAYER_NAMES


class Move(namedtuple("Move", ["player", "row", "col"])):
    def __repr__(self):
        return "Move(player={},row={},col={})".format(
            PLAYER_NAMES[self.player], self.row, self.col)


class Agent(ABC):
    def __init__(self, player):
        self._player = player

    @abstractmethod
    def next_move(self, board):
        pass


def valid_moves(board, player):
    return [Move(player, i, j)
            for i, j in board.empty_cells]
