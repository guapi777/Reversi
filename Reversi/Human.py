from .Player import Player


class Human(Player):
    def next_move(self, board):
        print("Player " + self.player_symbol + "'s turn.")
        reverse = "X" if self.player_symbol is "O" else "O"
        while True:
            step = str(input("Type the row and col to put the disc: "))
            pos = list(map(int, step.split(" ")))
            if self.if_legal_move(board, self.player_symbol, reverse, pos[0]-1, pos[1]-1):
                return [pos[0], pos[1]]
            print("Invalid Input")
