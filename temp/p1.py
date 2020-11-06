import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
dr = [0, 1, 1, 1, 0, -1, -1, -1]
dc = [-1, -1, 0, 1, 1, 1, 0, -1]
INF = 100000
MARGIN = 10000
START_TIME = 0
TIME_LIMIT = 3.5
WEIGHTS = np.array([
    [20, -3, 11, 8, 8, 11, -3, 20],
    [-3, -7, -4, 1, 1, -4, -7, -3],
    [11, -4, 2, 2, 2, 2, -4, 11],
    [8, 1, 2, -3, -3, 2, 1, 8],
    [8, 1, 2, -3, -3, 2, 1, 8],
    [11, -4, 2, 2, 2, 2, -4, 11],
    [-3, -7, -4, 1, 1, -4, -7, -3],
    [20, -3, 11, 8, 8, 11, -3, 20]
])


# don't change the class name
class AI(object):

    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need add your decision into your candidate_list. System will get the end of your candidate_list as your
        # decision.
        self.candidate_list = []

    # The input is current chessboard.
    def go(self, chessboard):

        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        # ==================================================================
        # Write your algorithm here
        # Here is the simplest sample:Random decision
        for col in range(8):
            for row in range(8):
                for i in range(8):
                    if (row + dr[i] < 8) & (row + dr[i] >= 0) & (col + dc[i] < 8) & (
                            col + dc[i] >= 0) & (chessboard[row][col] == 0):
                        r = row + dr[i]
                        c = col + dc[i]
                        if (chessboard[r][c] != 0) & (chessboard[r][c] != self.color):
                            self.next_search(r, c, i, chessboard, row, col)

        # ==============Find new pos========================================
        # Make sure that the position of your decision in chess board is empty.
        # If not, the system will return error.
        # Add your decision into candidate_list, Records the chess board
        # You need add all the positions which is valid
        # candidate_list example: [(3,3),(4,4)]
        # You need append your decision at the end of the candidate_list,
        # we will choose the last element of the candidate_list as the position you choose
        # If there is no valid position, you must return a empty list.40

    def next_search(self, r, c, i, chessboard, row, col):

        chess_size = len(chessboard)
        if (r + dr[i] < chess_size) & (r + dr[i] >= 0) & (c + dc[i] < chess_size) & (c + dc[i] >= 0):
            r = r + dr[i]
            c = c + dc[i]
            if (chessboard[r][c] != 0) & (chessboard[r][c] != self.color):
                self.next_search(r, c, i, chessboard, row, col)
            elif chessboard[r][c] == self.color:
                self.candidate_list.append((row, col))

    def alpha_beta(self, chessboard, color, depth, alpha, beta):
        best_action = -1, -1
        if self.is_game_ended():
            value = np.sum(chessboard)
            if color == 1:  # white
                value = -value
            value += MARGIN if value > 0 else -MARGIN
            return best_action, value

        if depth == 0 or is_timeout():
            value = np.sum(chessboard * WEIGHTS)
            if color == 1:
                value = -value
            return best_action, value

        board = chessboard
        color = color

        has_legal_move = False
        for i in range(8):
            for j in range(8):
                if board[i][j] != 0 or not self.place(i, j, color, check_only=True):
                    continue
                has_legal_move = True
                new_game = self.copy()
                new_game.apply_moveXY(i, j)
                value = -self.alpha_beta(new_game, depth - 1, -beta, -alpha)[1]
                if value > alpha:
                    alpha = value
                    best_action = i, j
                    if beta <= alpha:
                        return best_action, value

        if not has_legal_move:
            new_game = game.copy()
            new_game.apply_moveXY(-1, -1)
            return best_action, -alpha_beta(new_game, depth, -beta, -alpha)[1]

        return best_action, alpha

    def is_game_ended(self):
        pass

    def place(self, i, j, color, check_only):
        pass

    def copy(self):
        pass
