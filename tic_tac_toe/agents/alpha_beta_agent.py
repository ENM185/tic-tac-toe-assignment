from copy import deepcopy

from .base_agent import valid_moves
from .minimax_agent import Agent, valid_moves
from ..player import other_player
from ..board import CellState


class AlphaBetaAgent(Agent):
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

    def _minimax_move(self, move, board, player, depth=0, alpha=-200, beta=200):
        # update board for new "state"
        board.set_cell(move.player, move.row, move.col)
        self._states_visited_last_turn += 1
        score = self._minimax_state(board, player, depth+1, alpha, beta)
        board.set_cell(CellState.EMPTY, move.row, move.col)
        return score

    def _minimax_state(self, board, player, depth, alpha, beta):
        if hash(board) in self._cached_states:
            return self._cached_states[hash(board)]

        # check terminal cases
        evaluation = self._evaluate(board, depth)
        if type(evaluation) is tuple:
            return evaluation

        # recursive case
        multiplier = 1 if self._player == player else -1 #decide whether to use max or min
        score = (multiplier * -200, 0)
        for next_move in valid_moves(board, player):
            score = self._apply_multiplier(multiplier, max(
                self._apply_multiplier(multiplier, score), 
                self._apply_multiplier(multiplier, 
                    self._minimax_move(next_move, board, other_player(player),depth,-beta,-alpha))))
            if multiplier * score[0] >= beta:
                return score
            alpha = max(alpha, multiplier * score[0])
        self._cached_states[hash(board)] = score
        return score

    def _apply_multiplier(self, multiplier, score):
        return (multiplier * score[0], score[1])

    def _evaluate(self, board, depth):
        winner = board.winner
        if winner == self._player:
            return (100, depth)
        if winner == other_player(self._player):
            return (-100, depth)
        if not valid_moves(board, self._player):
            return (0, depth)
        return False