from .base_agent import valid_moves
from .minimax_agent import MinimaxAgent
from ..player import other_player


class AlphaBetaAgent(MinimaxAgent):
    def __init__(self, player):
        super().__init__(player)
        self._states_visited_last_turn = 0

    def next_move(self, board):
        self._states_visited_last_turn = 1 # this counts as a state
        return max(
            [(move,self._minimax(move, board, other_player(self._player), -2, 2)) for move in valid_moves(board, self._player)], 
            key=lambda t: t[1] # select min by score in (move, score)
            )[0] # select move in (move, score)

    @property
    def states_visited_last_turn(self):
        return self._states_visited_last_turn

    def _better_case(self, is_max_player, score, alphabeta):
        if is_max_player:
            if score >= alphabeta[1]:
                return True
            alphabeta[0] = max(alphabeta[0], score)
        else:
            if score <= alphabeta[0]:
                return True
            alphabeta[1] = min(alphabeta[1], score)
        return False