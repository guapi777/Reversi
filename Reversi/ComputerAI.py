from .Player import Player
import numpy as np
import copy


class ComputerAI(Player):
    weighting = [[500, -25, 10, 5, 5, 10, -25, 500],
                 [-25, -45, 1, 1, 1, 1, -45, -25],
                 [10, 1, 3, 2, 2, 3, 1, 10],
                 [5, 1, 2, 1, 1, 2, 1, 5],
                 [5, 1, 2, 1, 1, 2, 1, 5],
                 [10, 1, 3, 2, 2, 3, 1, 10],
                 [-25, -45, 1, 1, 1, 1, -45, -25],
                 [500, -25, 10, 5, 5, 10, -25, 500]]

    def __init__(self, symbol):
        super().__init__(symbol)
        self.reverse_player = "X" if symbol is "O" else "O"

    def out_of_range(self, i, j):
        if i < 0 or i > 7 or j < 0 or j > 7:
            return True
        return False

    def get_next_move(self, board, currentPlayer, reverse, r, c):
        if board[r][c] != ' ':
            return None

        is_legal = False

        for i in range(-1, 2):
            for j in range(-1, 2):
                step = 1
                while not self.out_of_range(r + i * step, c + j * step) and board[r + i * step][
                    c + j * step] == reverse:
                    step += 1
                if step > 1 and not self.out_of_range(r + i * step, c + j * step) \
                        and board[r + i * step][c + j * step] == currentPlayer:
                    is_legal = True
                    board[r][c] = currentPlayer
                    while True:
                        step = step - 1
                        board[(r + i * step)][c + j * step] = currentPlayer
                        if step <= 1:
                            break
        if is_legal:
            return board
        else:
            return None

    def get_weighting(self, board):
        weighting_sum = 0
        for r in range(8):
            for c in range(8):
                if board[r][c] == self.player_symbol:
                    weighting_sum += self.weighting[r][c]
                elif board[r][c] == self.reverse_player:
                    weighting_sum -= self.weighting[r][c]
        return weighting_sum

    def get_ratio(self, board):
        corner_c = [0, 7, 7, 0]
        corner_r = [0, 0, 7, 7]
        valid = [0, 0]
        invalid = [0, 0, 0, 0]
        edge_c = [1, 0, -1, 0]
        edge_r = [0, 1, 0, -1]

        for x in range(4):
            if self.player_symbol == board[corner_r[x]][corner_c[x]]:
                valid[0] += 1
                invalid[x] = 1
                for y in range(1, 7):
                    if self.player_symbol != board[y * edge_r[x] + corner_r[x]][y * edge_c[x] + corner_c[x]]:
                        break
                    else:
                        valid[1] += 1
                        invalid[x] = y + 1

        for x in range(4):
            if self.player_symbol == board[corner_r[x]][corner_c[x]]:
                for y in range(1, 7 - invalid[x - 1]):
                    break
                else:
                    valid[1] += 1

        return valid

    def calculate_ratio_and_weighting(self, board, available_move):
        ratio = self.get_ratio(board)
        reverse_available_move = self.find_reverse_move(board)
        weighting = int(sum(ratio) * 10 + self.get_weighting(board)
                        + (len(available_move) - len(reverse_available_move)) * 15)
        return weighting

    def alpha_beta_pruning(self, moving_player, board, alpha, beta, tree_depth, max_tree_depth):
        waiting_player = "X" if moving_player is "O" else "O"
        available_move, predict_board = self.find_available_move(moving_player, board)
        if len(available_move) == 0:
            return self.calculate_ratio_and_weighting(board, available_move), [-1, -1]
        if tree_depth == 0:
            return self.calculate_ratio_and_weighting(board, available_move), []

        if tree_depth == max_tree_depth:
            for r in range(len(available_move)):
                if self.player_symbol == moving_player and self.weighting[0][0] == self.weighting[available_move[r][0]]:
                    return 1000, available_move[r]

        if tree_depth >= 4:
            efficient_move = []
            for i in range(len(predict_board)):
                predict_score, predict_move = self.alpha_beta_pruning(waiting_player, predict_board[i], -10000, 10000,
                                                                      1, max_tree_depth)
                efficient_move.append(predict_score)
            sorted_score = np.argsort(efficient_move, kind="quicksort")
            available_move = [available_move[x] for x in sorted_score[0:5]]
            predict_board = [predict_board[x] for x in sorted_score[0:5]]

        highest_score = -10000
        best_move = []

        for i in range(len(predict_board)):
            next_score, next_move = self.alpha_beta_pruning(waiting_player, predict_board[i], -beta,
                                                            -max(alpha, highest_score), tree_depth - 1, max_tree_depth)
            next_score = -next_score

            if highest_score < next_score:
                highest_score = next_score
                best_move = available_move[i]
                if beta < highest_score:
                    return highest_score, best_move

        return highest_score, best_move

    def check_reverse_legal_move(self, board, r, c):
        if board[r][c] != ' ':
            return False

        for i in range(-1, 2):
            for j in range(-1, 2):
                step = 1
                while not self.out_of_range(r + step * i, c + step * j) and self.player_symbol \
                        == board[r + i * step][c + j * step]:
                    step += 1
                if step > 1 and not self.out_of_range(r + step * i, c + step * j) \
                        and self.reverse_player == board[r + step * i][c + step * j]:
                    return True

        return False

    def find_available_move(self, current_player, board):
        reverse_player = "X" if current_player is "O" else "O"
        available_move = []
        predict_board = []

        for i in range(0, 8):
            for j in range(0, 8):
                temp_board = copy.deepcopy(board)
                next_board = self.get_next_move(temp_board, current_player, reverse_player, i, j)
                if next_board is not None:
                    available_move.append([i, j])
                    predict_board.append(next_board)
        return available_move, predict_board

    def find_reverse_move(self, board):
        available_move = []

        for i in range(0, 8):
            for j in range(0, 8):
                next_board = self.check_reverse_legal_move(board, i, j)
                if next_board:
                    available_move.append([i, j])
        return available_move

    def next_move(self, board):
        print("Player " + self.player_symbol + "'s turn.")
        print("Type the row and col to put the disc: ", end="")
        best_score, best_move = self.alpha_beta_pruning(self.player_symbol, board, -10000, -10000,
                                                        7, 7)  # tuning the depth
        print(best_move[0] + 1, best_move[1] + 1)
        return [best_move[0] + 1, best_move[1] + 1]
