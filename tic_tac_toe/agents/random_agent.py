import random

from .base_agent import Agent, valid_moves


class RandomAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def next_move(self, board):
        return random.choice(valid_moves(board, self._player))
