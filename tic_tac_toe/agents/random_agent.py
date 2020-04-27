import random

from .base_agent import Agent


class RandomAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def next_move(self, board):
        pass
