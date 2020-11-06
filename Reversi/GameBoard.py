class GameBoard:
    def __init__(self):
        self.board = None

    def init_game_board(self):
        self.board = []
        for i in range(8):
            self.board.append([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])
        self.board[3][4] = "O"
        self.board[4][3] = "O"
        self.board[3][3] = "X"
        self.board[4][4] = "X"

    def check_ending(self):
        player1_end = self.check_legal_move("O")
        player2_end = self.check_legal_move("X")

        if player1_end or player2_end:
            return False
        else:
            print("There is no valid move for both player. End Game!")
            return True

    def out_of_range(self, i, j):
        if i < 0 or i > 7 or j < 0 or j > 7:
            return True
        return False

    def if_legal_move(self, current_player, reverse, r, c, need_flip):
        is_legal = False
        if self.board[r][c] != ' ':
            return False

        for i in range(-1, 2):
            for j in range(-1, 2):
                step = 1
                while not GameBoard.out_of_range(self, r + i * step, c + j * step) \
                        and self.board[r + i * step][c + j * step] == reverse:
                    step = step + 1

                if step > 1 and not GameBoard.out_of_range(self, r + i * step, c + j * step) \
                        and self.board[r + i * step][c + j * step] == current_player:
                    is_legal = True
                    if need_flip:
                        self.board[r][c] = current_player
                        while True:
                            step = step - 1
                            self.board[(r + i * step)][c + j * step] = current_player
                            if step <= 1:
                                break
        return is_legal

    def check_legal_move(self, symbol):
        reverse = ['O', 'X'][symbol == 'O']
        for r in range(0, 8):
            for c in range(0, 8):
                if self.board[r][c] == ' ' and self.if_legal_move(symbol, reverse, r, c, False):
                    return True
        return False

    def check_winner(self):
        chess_counter = [0, 0]
        for r in range(0, 8):
            for c in range(0, 8):
                if self.board[r][c] == 'O':
                    chess_counter[0] = chess_counter[0] + 1
                elif self.board[r][c] == 'X':
                    chess_counter[1] = chess_counter[1] + 1
        return chess_counter

    def execute_flip(self, pos, symbol):
        pos = [pos[0] - 1, pos[1] - 1]
        reverse = ['O', 'X'][symbol == 'O']
        self.if_legal_move(symbol, reverse, pos[0], pos[1], True)

    def print_game_board(self):
        print("   | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |")
        print("------------------------------------")
        for i in range(8):
            print(" " + str(i + 1) + " | ", end="")
            print(*self.board[i][0:8], sep=" | ", end=" |\n")
            print("------------------------------------")
