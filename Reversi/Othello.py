from .GameBoard import GameBoard
from .Human import Human
from .ComputerAI import ComputerAI


class Othello:
    def __init__(self):
        self.gameBoard = GameBoard()
        self.player1 = None
        self.player2 = None
        self.turn = 0

    def create_player(self, symbol, player_num):
        print("Please choose player " + str(player_num) + " (" + symbol + "):")
        print("1. Human")
        print("2. Computer Player")
        player_type = int(input("Your Choice is: "))

        if player_type == 1:
            print("Player " + symbol + " is " + 'Human.')
            return Human(symbol)
        else:
            print("Player " + symbol + " is " + 'Computer(AI).')
            return ComputerAI(symbol)

    def start_game(self):
        self.player1 = self.create_player('O', 1)
        self.player2 = self.create_player('X', 2)
        self.gameBoard.init_game_board()
        self.gameBoard.print_game_board()
        while not self.gameBoard.check_ending():
            current_player = [self.player1, self.player2][self.turn]

            if self.gameBoard.check_legal_move(current_player.player_symbol):
                pos = current_player.next_move(self.gameBoard.board)
                self.gameBoard.execute_flip(pos, current_player.player_symbol)
            else:
                print("There is no valid move for Player " + current_player.player_symbol + ".")
            self.turn = 1 - self.turn

            self.gameBoard.print_game_board()

        s1, s2 = self.gameBoard.check_winner()
        if s1 > s2:
            winner = 'O'  # Black
        elif s1 < s2:
            winner = 'X'  # White
        elif s1 == s2:
            winner = ' '  # Tie

        print('Count O : {}'.format(s1))
        print('Count X : {}'.format(s2))
        if winner != ' ':
            print('Player {} won!\n'.format(winner))
        else:
            print('A tie')


if __name__ == "__main__":
    othello = Othello()
    othello.start_game()
