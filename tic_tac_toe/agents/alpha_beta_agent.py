from copy import deepcopy

from .base_agent import valid_moves
from .minimax_agent import Agent, valid_moves
from ..player import other_player
from ..board import CellState

import math

class AlphaBetaAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
        self._states_visited_last_turn = 0
        self._cached_leaf_nodes = {} #cache for fully expanded paths (up to max_depth for that iteration). Must clear each iteration
        self._eval_cache = {} #cache for heuristic (not as important)
        self._last_iter_cache = {} #cache found for most recent iteration (sometimes includes current iteration)

    def next_move(self, board):
        self._states_visited_last_turn = 1 # this counts as a state
        move = self._minimax(board, self._player, max_depth=None)[1]
        return move

    @property
    def states_visited_last_turn(self):
        return self._states_visited_last_turn

    def _minimax(self, board, player, depth=None, alpha=(-math.inf,0), beta=(math.inf,0), pruned=[False], max_depth = None):
        moves = valid_moves(board, player)
        moves.sort(key=lambda move: tuple([x for x in self._last_iter_cache[hash(board)]]) if hash(board) in self._last_iter_cache else self._evaluate_move(board, player, move))

        if depth is None:
            depth = board.size**2 - len(moves)
            for move in moves:
                print(move)
                print(tuple([x for x in self._last_iter_cache[hash(board)]]) if hash(board) in self._last_iter_cache else None)
                print(self._evaluate_move(board,player,move))

        (score, move) = ((-math.inf,0), None)
        for next_move in moves:
            child_pruned = [False] #every child by default is not pruned, so that siblings don't affect each other

            value = self._minimax_move(next_move, board, other_player(player), depth+1, (-beta[0],-beta[1]), (-alpha[0],-alpha[1]), child_pruned, max_depth)
            value = (-value[0],-value[1]) #negamax

            if value > score:
                (score, move) = (value, next_move) #max value

            if child_pruned[0]:
                pruned[0] = True #anything with pruned descendants is pruned

            if score >= beta:
                pruned[0] = True #where pruning originates
                return (score, move)
            alpha = max(score, alpha)

        if not pruned[0]:
            self._cached_leaf_nodes[hash(board)] = score

        return (score, move)

    def _minimax_move(self, move, board, player, depth, alpha, beta, pruned, max_depth):
        # update board for new "state"
        board.set_cell(move.player, move.row, move.col)

        self._states_visited_last_turn += 1
        score = self._minimax_state(board, player, depth, alpha, beta, pruned, max_depth)
        self._last_iter_cache[hash(board)] = score

        board.set_cell(CellState.EMPTY, move.row, move.col)
        return score

    def _minimax_state(self, board, player, depth, alpha, beta, pruned, max_depth):
        if hash(board) in self._cached_leaf_nodes:
            return self._cached_leaf_nodes[hash(board)]

        # check terminal cases
        winner = board.winner
        if winner is not None:
            return (-math.inf, depth) #player before this just won
        if not valid_moves(board, player):
            return (0, depth)

        if depth == max_depth:
            score = (self._evaluate_state(board, player), depth)
            return score

        score = self._minimax(board, player, depth, alpha, beta, pruned, max_depth)[0]
        return score

    def _evaluate_move(self, board, player, move):
        board.set_cell(move.player, move.row, move.col)
        score = self._evaluate_state(board, player)
        self._eval_cache[hash(board)] = score
        board.set_cell(CellState.EMPTY, move.row, move.col)
        return score

    def _evaluate_state(self, board, player):
        # check leaf node cache
        if hash(board) in self._cached_leaf_nodes:
            return self._cached_leaf_nodes[hash(board)][0] * math.inf * (-1 if player != self._player else 1)
        if hash(board) in self._eval_cache:
            return self._eval_cache[hash(board)]

        winner = board.winner
        if winner == player:
            return math.inf
        if winner == other_player(player):
            return -math.inf
        if not valid_moves(board, player):
            return 0
        
        #not terminal case
        score = 0
        forks = [0,0] # amount of lines for (other, current) that are one move away from winning (2+ is a fork)
        for line in board.all_lines:
            #use sublines of length num_to_win for cases where this is < board
            for limit in range(board.size-board.num_to_win+1):
                subline = line[limit:limit+board.num_to_win]
                count = [0,0] # (other, current)
                for cell in subline:
                    if cell == player:
                        count[1] += 1
                    elif cell == other_player(player):
                        count[0] += 1
                if count[0] == 0:
                    score += count[1]
                elif count[1] == 0:
                    score -= count[0]
                for i in range(2):
                    if count[i] == board.num_to_win - 1:
                        forks[i] += 1
        if forks[0] > 0:
            return -math.inf
        elif forks[1] > 1:
            return math.inf
        return score