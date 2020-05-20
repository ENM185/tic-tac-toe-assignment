from copy import deepcopy

from .base_agent import valid_moves
from .minimax_agent import Agent, valid_moves
from ..player import other_player
from ..board import CellState


class AlphaBetaAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
        self._states_visited_last_turn = 0
        self._cached_leaf_nodes = {}
        self._eval_cache = {}

    def next_move(self, board):
        self._states_visited_last_turn = 1 # this counts as a state
        return max(
            [(move,self._minimax_move(move, board, other_player(self._player))) for move in valid_moves(board, self._player)], 
            key=lambda t: t[1] # select min by score in (move, score)
            )[0] # select move in (move, score)

    @property
    def states_visited_last_turn(self):
        return self._states_visited_last_turn

    def _minimax_move(self, move, board, player, depth=0, alpha=-2, beta=2):
        # update board for new "state"
        board.set_cell(move.player, move.row, move.col)
        self._states_visited_last_turn += 1
        score = self._minimax_state(board, player, depth+1, alpha, beta)
        board.set_cell(CellState.EMPTY, move.row, move.col)
        return score

    def _minimax_state(self, board, player, depth, alpha, beta):
        if hash(board) in self._cached_leaf_nodes:
            return self._cached_leaf_nodes[hash(board)]

        # check terminal cases
        winner = board.winner
        if winner == self._player:
            return (1, depth)
        if winner == other_player(self._player):
            return (-1, depth)
        if not valid_moves(board, self._player):
            return (0, depth)

        # recursive case
        multiplier = 1 if self._player == player else -1 #decide whether to use max or min
        score = (multiplier * -2, 0)

        moves = valid_moves(board, player)
        moves.sort(key=lambda move: -self._evaluate_move(board, player, move))

        for next_move in moves:

            score = self._apply_multiplier(multiplier, max(
                self._apply_multiplier(multiplier, score), 
                self._apply_multiplier(multiplier, 
                    self._minimax_move(next_move, board, other_player(player),depth,-beta,-alpha))))

            if multiplier * score[0] >= beta:
                return score
            alpha = max(alpha, multiplier * score[0])

        self._cached_leaf_nodes[hash(board)] = score
        return score

    def _apply_multiplier(self, multiplier, score):
        return (multiplier * score[0], score[1])

    def _evaluate_move(self, board, player, move):
        board.set_cell(move.player, move.row, move.col)
        score = self._evaluate(board, player)
        self._eval_cache[hash(board)] = score
        board.set_cell(CellState.EMPTY, move.row, move.col)
        return score

    def _evaluate(self, board, player):
        # check leaf node cache
        if hash(board) in self._cached_leaf_nodes:
            return self._cached_leaf_nodes[hash(board)][0] * 10000 * (-1 if player != self._player else 1)
        if hash(board) in self._eval_cache:
            return self._eval_cache[hash(board)]

        winner = board.winner
        if winner == player:
            return 10000
        if winner == other_player(player):
            return -10000
        if not valid_moves(board, player):
            return 0
        
        #not terminal case
        score = 0
        twos = [0,0] # forks: amount of lines for (other, current) that are one move away from winning
        for line in board.all_lines:
            count = [0,0] # (other, current)
            for cell in line:
                if cell == player:
                    count[1] += 1
                elif cell == other_player(player):
                    count[0] += 1
            if count[0] == 0:
                score += 1
            elif count[1] == 0:
                score -= 1
        if twos[0] > 0:
            return -10000
        if twos[1] == 1:
            score += 1
        elif twos[1] == -1:
            return 10000
        return score