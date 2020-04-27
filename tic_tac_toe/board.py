import numpy as np

from .player import Player, PLAYER_NAMES


class CellState:
    EMPTY = -1
    X = Player.X
    O = Player.O

    ALL_STATES = Player.ALL_PLAYERS + [EMPTY]

    CELL_CHAR = {
        EMPTY: " ",
        X: PLAYER_NAMES[X],
        O: PLAYER_NAMES[O],
    }


class Board(object):
    def __init__(self, size=3, num_to_win=None):
        num_to_win = num_to_win or size
        if num_to_win > size:
            raise ValueError("num_to_win cannot be larger than size.")

        self._size = size
        self._num_to_win = num_to_win
        self._board = CellState.EMPTY * np.ones(shape=(size, size),
                                                dtype=np.int8)

    def row(self, row_num):
        if row_num < 0 or row_num >= self._size:
            raise ValueError("row_num must be between 0 and {}.".format(
                self._size))

        return self._board[row_num, :]

    def col(self, col_num):
        if col_num < 0 or col_num >= self._size:
            raise ValueError("col_num must be between 0 and {}.".format(
                self._size))

        return self._board[:, col_num]

    def cell(self, row_num, col_num):
        if row_num < 0 or row_num >= self._size:
            raise ValueError("row_num must be between 0 and {}.".format(
                self._size))

        if col_num < 0 or col_num >= self._size:
            raise ValueError("col_num must be between 0 and {}.".format(
                self._size))

        return self._board[row_num, col_num]

    def main_diagonal(self, offset=0):
        return np.diagonal(self._board, offset=offset)

    def secondary_diagonal(self, offset=0):
        n = self._size - 1
        row_start = max(offset, 0)
        row_stop = min(n + offset, n)
        start = n * row_start + n + offset
        stop = n * row_stop + n + offset + 1
        step = n
        return self._board.ravel()[start:stop:step]

    def set_cell(self, state, row_num, col_num):
        if state not in CellState.ALL_STATES:
            raise ValueError("Cell state cannot be {}.".format(state))

        if row_num < 0 or row_num >= self._size:
            raise ValueError("row_num must be between 0 and {}.".format(
                self._size))

        if col_num < 0 or col_num >= self._size:
            raise ValueError("col_num must be between 0 and {}.".format(
                self._size))

        self._board[row_num, col_num] = state

        return self

    @property
    def size(self):
        return self._size

    @property
    def num_to_win(self):
        return self._num_to_win

    @property
    def diagonals(self):
        max_offset = self._size - self._num_to_win
        offsets = range(-max_offset, max_offset + 1)

        main_diagonals = [self.main_diagonal(offset) for offset in offsets]
        secondary_diagonals = [self.secondary_diagonal(offset)
                               for offset in offsets]
        return main_diagonals + secondary_diagonals

    @property
    def rows(self):
        return [self.row(i) for i in range(self._size)]

    @property
    def cols(self):
        return [self.col(i) for i in range(self._size)]

    @property
    def all_lines(self):
        return self.diagonals + self.rows + self.cols

    @property
    def empty_cells(self):
        return [(i, j)
                for i in range(self._size)
                for j in range(self._size)
                if self.cell(i, j) == CellState.EMPTY]

    @property
    def winner(self):
        def _line_winner(line):
            # scan through the line, counting the current longest sequence of
            # equal elements
            # when the length of the current sequence is sufficient for a win
            # and the current sequence is one of the players, return
            current_state = line[0]
            current_length = 1
            for i in range(1, len(line)):
                if current_state == line[i]:
                    current_length += 1
                else:
                    current_state = line[i]
                    current_length = 1

                if (current_length == self._num_to_win
                        and current_state in Player.ALL_PLAYERS):
                    return current_state

            return None

        for l in self.all_lines:
            line_winner = _line_winner(l)
            if line_winner is not None:
                return line_winner

        return None

    def __repr__(self):
        def _row_to_str(enumerated_row):
            i, row = enumerated_row
            return "{: >2}   ".format(i) \
                   + " │ ".join(map(lambda c: CellState.CELL_CHAR[c], row)) \
                   + " "

        row_separator = "\n    " + "┼".join(["───"] * self._size) + "\n"
        all_rows = row_separator.join(map(_row_to_str, enumerate(self.rows)))
        header = "    " + " ".join(map(lambda i: " {: <2}".format(i),
                                       range(self._size)))

        return header + "\n\n" + all_rows
