class Player:
    def __init__(self, symbol):
        self.player_symbol = symbol

    def out_of_range(self, i, j):
        if i < 0 or i > 7 or j < 0 or j > 7:
            return True
        return False

    def if_legal_move(self, board, current_player, reverse, r, c):
        if board[r][c] != ' ':
            return False

        for i in range(-1, 2):
            for j in range(-1, 2):
                step = 1
                while not Player.out_of_range(self, r + i * step, c + j * step) \
                        and board[r + i * step][c + j * step] == reverse:
                    step += 1
                if not Player.out_of_range(self, r + i * step, c + j * step) and step > 1 \
                        and board[r + i * step][c + j * step] == current_player:
                    return True
        return False

    def next_move(self, board):
        pass
