from pprint import pprint
import time

from .base_agent import Agent
from .minimax_agent import MinimaxAgent
from ..player import PLAYER_NAMES

class TimedAgent(Agent):
    def __init__(self, agent, board_size, stats):
        self._agent = agent
        self._runtime_per_turn = [0] * (board_size ** 2)
        self._states_visited_per_turn = [0] * (board_size ** 2)
        self._moves_per_turn = [0] * (board_size ** 2)
        self._stats = stats

    def next_move(self, board):
        current_turn = board.size ** 2 - len(board.empty_cells)

        start = time.process_time()
        move = self._agent.next_move(board)
        end = time.process_time()

        if self._stats:
            self._stats['turn'].append(current_turn+1)
            self._stats['runtime'].append(end - start)
            try:
                states = self._agent.states_visited_last_turn
            except:
                states = None
            self._stats['states_visited'].append(states)

        return move

    def display_stats(self):
        print("\tAverage Runtimes (in seconds):"
            + str(["%5.2e" % num for num in self._average_per_turn(self._runtime_per_turn)]))
        if self._track_states:
            print("\tAverage States Visited:"
            + str(["%6d" % num for num in self._average_per_turn(self._states_visited_per_turn)]))

    def _average_per_turn(self, _list):
        return [_list[i]/self._moves_per_turn[i] if self._moves_per_turn[i] != 0 else 0 for i in range(len(_list))]