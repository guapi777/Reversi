import numpy as np

chessboard = [[1, 0, 1, 0, -1, -1, 1, 1],
              [1, 1, 1, -1, -1, 1, 1, 1],
              [1, 1, -1, -1, -1, 1, 1, 1],
              [1, 1, -1, 1, -1, 1, 1, 1],
              [1, -1, 1, -1, 1, 1, 1, 1],
              [1, 1, 1, 1, -1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1],
              [1, 0, 1, 0, 0, 1, 0, 1]]





def eval_stable(chessboard, color):
    new_chessboard = np.zeros((8, 8))
    # 左上
    if chessboard[0][0] != 0:
        now_color = chessboard[0][0]
        len = 8
        for i in range(8):
            for j in range(len):
                if chessboard[i][j] == now_color:
                    new_chessboard[i][j] = now_color
                else:
                    len = j
                    break

    # 右上
    if chessboard[0][7] != 0:
        now_color = chessboard[0][7]
        len = -1
        for i in range(8):
            for j in range(7, len, -1):
                if chessboard[i][j] == now_color:
                    new_chessboard[i][j] = now_color
                else:
                    len = j
                    break
    # 左下
    if chessboard[7][0] != 0:
        now_color = chessboard[7][0]
        len = 8
        for i in range(7, -1, -1):
            for j in range(len):
                if chessboard[i][j] == now_color:
                    new_chessboard[i][j] = now_color
                else:
                    len = j
                    break
    # 右下
    if chessboard[7][7] != 0:
        now_color = chessboard[7][7]
        len = -1
        for i in range(7, -1, -1):
            for j in range(7, len, -1):
                if chessboard[i][j] == now_color:
                    new_chessboard[i][j] = now_color
                else:
                    len = j
                    break

    value = np.sum(new_chessboard)
    if color == -1:
        value = -value
    return value


print(eval_stable(chessboard, -1))
