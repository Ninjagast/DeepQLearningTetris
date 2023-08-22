import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.pyplot import draw
import time
import tensorflow as tf

class TetrisBoard:
    def __init__(self, training=True):
        self.board = np.zeros((10, 20), dtype=np.int32)
        self.pieces = [0, 1, 2, 3, 4, 5, 6]
        self.grabbed = -1
        self.dead = False
        self.points = 0
        # random.shuffle(self.pieces)
        self.lines_cleared = 0
        self.training = training
        self.pieces_placed = 0

        self.o_color = 1
        self.l_color = 2
        self.s_color = 3
        self.z_color = 4
        self.i_color = 5
        self.j_color = 6
        self.t_color = 7

    def get_new_piece(self) -> int:
        self.grabbed += 1

        if self.grabbed == len(self.pieces):
            self.grabbed = 0
            random.shuffle(self.pieces)

        return self.pieces[self.grabbed]

    def step(self, action, piece) -> tuple:
        start_points = self.points

        data = action / 10
        rotation = int(data)
        column = round(((data - int(data)) * 10))
        self.place_piece(piece, column, rotation)

        new_piece = self.get_new_piece()
        return_arr = np.zeros(7)
        return_arr[new_piece] = 1

        if self.lines_cleared == 40:
            self.points += 10000
            self.dead = True

        return_board = np.zeros(10, dtype=np.float32)

        for i in range(10):
            return_board[i] = self.get_highest_point(i)

        if self.dead:
            self.points -= 10

        move_points = self.points - start_points
        return (np.concatenate((return_arr, return_board), axis=None, dtype=np.float32), new_piece), move_points, self.dead, ""

    def get_cleared_lines(self):
        return self.lines_cleared

    # rotation is in 0,1,2,3 based on png
    # https://static.wikia.nocookie.net/tetrisconcept/images/3/3d/SRS-pieces.png/revision/latest?cb=20060626173148
    def place_piece(self, piece: str, column1: int, rotation: int) -> bool:
        self.pieces_placed += 1
        not_found = True
        created_gap = False
        first_column_highest = 0
        second_column_highest = 0
        third_column_highest = 0
        fourth_column_highest = 0

        # O / square piece
        if piece == 0:
            column2 = column1 + 1

            if column1 == 9:
                column2 = column1 - 1

            first_column_highest = self.get_highest_point(column1)
            second_column_highest = self.get_highest_point(column2)

            if self.dead:
                return False

            if not first_column_highest == second_column_highest:
                created_gap = True

            if first_column_highest >= second_column_highest:
                self.board[column1][first_column_highest] = self.o_color
                self.board[column2][first_column_highest] = self.o_color
                self.board[column1][first_column_highest + 1] = self.o_color
                self.board[column2][first_column_highest + 1] = self.o_color
            else:
                self.board[column1][second_column_highest] = self.o_color
                self.board[column2][second_column_highest] = self.o_color
                self.board[column1][second_column_highest + 1] = self.o_color
                self.board[column2][second_column_highest + 1] = self.o_color

        # I / long piece
        if piece == 1:
            if rotation == 0 or rotation == 2:
                if column1 > 6:
                    column1 = 6
                    column2 = 7
                    column3 = 8
                    column4 = 9
                else:
                    column2 = column1 + 1
                    column3 = column1 + 2
                    column4 = column1 + 3

                first_column_highest = self.get_highest_point(column1)
                second_column_highest = self.get_highest_point(column2)
                third_column_highest = self.get_highest_point(column3)
                fourth_column_highest = self.get_highest_point(column4)

                if self.dead:
                    return False

                if not first_column_highest == second_column_highest == third_column_highest == fourth_column_highest:
                    created_gap = True

                if first_column_highest >= second_column_highest and first_column_highest >= third_column_highest and first_column_highest >= fourth_column_highest:
                    self.board[column1][first_column_highest] = self.i_color
                    self.board[column2][first_column_highest] = self.i_color
                    self.board[column3][first_column_highest] = self.i_color
                    self.board[column4][first_column_highest] = self.i_color

                elif second_column_highest >= first_column_highest and second_column_highest >= third_column_highest and second_column_highest >= fourth_column_highest:
                    self.board[column1][second_column_highest] = self.i_color
                    self.board[column2][second_column_highest] = self.i_color
                    self.board[column3][second_column_highest] = self.i_color
                    self.board[column4][second_column_highest] = self.i_color

                elif third_column_highest >= first_column_highest and third_column_highest >= second_column_highest and third_column_highest >= fourth_column_highest:
                    self.board[column1][third_column_highest] = self.i_color
                    self.board[column2][third_column_highest] = self.i_color
                    self.board[column3][third_column_highest] = self.i_color
                    self.board[column4][third_column_highest] = self.i_color

                elif fourth_column_highest >= first_column_highest and fourth_column_highest >= second_column_highest and fourth_column_highest >= third_column_highest:
                    self.board[column1][fourth_column_highest] = self.i_color
                    self.board[column2][fourth_column_highest] = self.i_color
                    self.board[column3][fourth_column_highest] = self.i_color
                    self.board[column4][fourth_column_highest] = self.i_color

            else:
                first_column_highest = self.get_highest_point(column1)

                if self.dead:
                    return False

                self.board[column1][first_column_highest] = self.i_color
                self.board[column1][first_column_highest + 1] = self.i_color
                self.board[column1][first_column_highest + 2] = self.i_color
                self.board[column1][first_column_highest + 3] = self.i_color

        # S piece
        if piece == 2:
            if rotation == 0 or rotation == 2:
                if column1 > 7:
                    column1 = 7
                    column2 = 8
                    column3 = 9
                else:
                    column2 = column1 + 1
                    column3 = column1 + 2

                first_column_highest = self.get_highest_point(column1)
                second_column_highest = self.get_highest_point(column2)
                third_column_highest = self.get_highest_point(column3)

                if self.dead:
                    return False

                if not first_column_highest == second_column_highest or not third_column_highest - second_column_highest == 1:
                    created_gap = True

                if first_column_highest >= second_column_highest and first_column_highest >= third_column_highest:
                    self.board[column1][first_column_highest] = self.s_color
                    self.board[column2][first_column_highest] = self.s_color
                    self.board[column2][first_column_highest + 1] = self.s_color
                    self.board[column3][first_column_highest + 1] = self.s_color
                elif second_column_highest >= first_column_highest and second_column_highest >= third_column_highest:
                    self.board[column1][second_column_highest] = self.s_color
                    self.board[column2][second_column_highest] = self.s_color
                    self.board[column2][second_column_highest + 1] = self.s_color
                    self.board[column3][second_column_highest + 1] = self.s_color
                elif third_column_highest >= first_column_highest and third_column_highest >= second_column_highest:
                    self.board[column1][third_column_highest - 1] = self.s_color
                    self.board[column2][third_column_highest - 1] = self.s_color
                    self.board[column2][third_column_highest] = self.s_color
                    self.board[column3][third_column_highest] = self.s_color

            else:
                if column1 == 9:
                    column1 = 8
                    column2 = 9
                else:
                    column2 = column1 + 1

                first_column_highest = self.get_highest_point(column1)
                second_column_highest = self.get_highest_point(column2)

                if self.dead:
                    return False

                if not first_column_highest - second_column_highest == 1:
                    created_gap = True

                if second_column_highest >= first_column_highest:
                    self.board[column2][second_column_highest] = self.s_color
                    self.board[column2][second_column_highest + 1] = self.s_color
                    self.board[column1][second_column_highest + 1] = self.s_color
                    self.board[column1][second_column_highest + 2] = self.s_color

                else:
                    self.board[column1][first_column_highest] = self.s_color
                    self.board[column1][first_column_highest + 1] = self.s_color
                    self.board[column2][first_column_highest] = self.s_color
                    self.board[column2][first_column_highest - 1] = self.s_color

        # Z piece
        if piece == 3:
            if rotation == 0 or rotation == 2:
                if column1 > 7:
                    column1 = 7
                    column2 = 8
                    column3 = 9
                else:
                    column2 = column1 + 1
                    column3 = column1 + 2

                first_column_highest = self.get_highest_point(column1)
                second_column_highest = self.get_highest_point(column2)
                third_column_highest = self.get_highest_point(column3)

                if self.dead:
                    return False

                if not second_column_highest == third_column_highest or not first_column_highest - second_column_highest == 1:
                    created_gap = True

                if second_column_highest >= third_column_highest and second_column_highest >= first_column_highest:
                    self.board[column1][second_column_highest + 1] = self.z_color
                    self.board[column2][second_column_highest] = self.z_color
                    self.board[column2][second_column_highest + 1] = self.z_color
                    self.board[column3][second_column_highest] = self.z_color

                elif third_column_highest >= second_column_highest and third_column_highest >= first_column_highest:
                    self.board[column1][third_column_highest + 1] = self.z_color
                    self.board[column2][third_column_highest] = self.z_color
                    self.board[column2][third_column_highest + 1] = self.z_color
                    self.board[column3][third_column_highest] = self.z_color

                elif first_column_highest >= second_column_highest and first_column_highest >= third_column_highest:
                    self.board[column1][first_column_highest] = self.z_color
                    self.board[column2][first_column_highest] = self.z_color
                    self.board[column2][first_column_highest - 1] = self.z_color
                    self.board[column3][first_column_highest - 1] = self.z_color

            else:
                if column1 == 9:
                    column1 = 8
                    column2 = 9
                else:
                    column2 = column1 + 1

                first_column_highest = self.get_highest_point(column1)
                second_column_highest = self.get_highest_point(column2)

                if self.dead:
                    return False

                if not second_column_highest - first_column_highest == 1:
                    created_gap = True

                if first_column_highest >= second_column_highest:
                    self.board[column1][first_column_highest] = self.z_color
                    self.board[column1][first_column_highest + 1] = self.z_color
                    self.board[column2][first_column_highest + 1] = self.z_color
                    self.board[column2][first_column_highest + 2] = self.z_color
                else:
                    self.board[column2][second_column_highest] = self.z_color
                    self.board[column2][second_column_highest + 1] = self.z_color
                    self.board[column1][second_column_highest] = self.z_color
                    self.board[column1][second_column_highest - 1] = self.z_color

        # L piece
        if piece == 4:
            if rotation == 0:
                if column1 > 7:
                    column1 = 7
                    column2 = 8
                    column3 = 9
                else:
                    column2 = column1 + 1
                    column3 = column1 + 2

                first_column_highest = self.get_highest_point(column1)
                second_column_highest = self.get_highest_point(column2)
                third_column_highest = self.get_highest_point(column3)

                if self.dead:
                    return False

                if not first_column_highest == second_column_highest == third_column_highest:
                    created_gap = True

                if first_column_highest >= second_column_highest and first_column_highest >= third_column_highest:
                    self.board[column1][first_column_highest] = self.l_color
                    self.board[column2][first_column_highest] = self.l_color
                    self.board[column3][first_column_highest] = self.l_color
                    self.board[column3][first_column_highest + 1] = self.l_color

                elif second_column_highest >= first_column_highest and second_column_highest >= third_column_highest:
                    self.board[column1][second_column_highest] = self.l_color
                    self.board[column2][second_column_highest] = self.l_color
                    self.board[column3][second_column_highest] = self.l_color
                    self.board[column3][second_column_highest + 1] = self.l_color
                else:
                    self.board[column1][third_column_highest] = self.l_color
                    self.board[column2][third_column_highest] = self.l_color
                    self.board[column3][third_column_highest] = self.l_color
                    self.board[column3][third_column_highest + 1] = self.l_color

            elif rotation == 1:
                if column1 == 9:
                    column1 = 8
                    column2 = 9
                else:
                    column2 = column1 + 1

                first_column_highest = self.get_highest_point(column1)
                second_column_highest = self.get_highest_point(column2)

                if self.dead:
                    return False

                if not first_column_highest == second_column_highest:
                    created_gap = True

                if first_column_highest >= second_column_highest:
                    self.board[column1][first_column_highest] = self.l_color
                    self.board[column1][first_column_highest + 1] = self.l_color
                    self.board[column1][first_column_highest + 2] = self.l_color
                    self.board[column2][first_column_highest] = self.l_color

                else:
                    self.board[column1][second_column_highest] = self.l_color
                    self.board[column1][second_column_highest + 1] = self.l_color
                    self.board[column1][second_column_highest + 2] = self.l_color
                    self.board[column2][second_column_highest] = self.l_color

            elif rotation == 2:
                if column1 > 7:
                    column1 = 7
                    column2 = 8
                    column3 = 9
                else:
                    column2 = column1 + 1
                    column3 = column1 + 2

                first_column_highest = self.get_highest_point(column1)
                second_column_highest = self.get_highest_point(column2)
                third_column_highest = self.get_highest_point(column3)

                if self.dead:
                    return False

                if not second_column_highest - first_column_highest == 1 or not second_column_highest == third_column_highest:
                    created_gap = True

                if first_column_highest >= second_column_highest and first_column_highest >= third_column_highest:
                    self.board[column1][first_column_highest] = self.l_color
                    self.board[column1][first_column_highest + 1] = self.l_color
                    self.board[column2][first_column_highest + 1] = self.l_color
                    self.board[column3][first_column_highest + 1] = self.l_color

                elif second_column_highest >= first_column_highest and second_column_highest >= third_column_highest:
                    self.board[column1][second_column_highest - 1] = self.l_color
                    self.board[column1][second_column_highest] = self.l_color
                    self.board[column2][second_column_highest] = self.l_color
                    self.board[column3][second_column_highest] = self.l_color
                else:
                    self.board[column1][third_column_highest - 1] = self.l_color
                    self.board[column1][third_column_highest] = self.l_color
                    self.board[column2][third_column_highest] = self.l_color
                    self.board[column3][third_column_highest] = self.l_color

            else:
                if column1 == 9:
                    column1 = 8
                    column2 = 9
                else:
                    column2 = column1 + 1

                first_column_highest = self.get_highest_point(column1)
                second_column_highest = self.get_highest_point(column2)

                if not first_column_highest - second_column_highest == 2:
                    created_gap = True

                if self.dead:
                    return False

                if second_column_highest >= first_column_highest or second_column_highest + 1 == first_column_highest:
                    self.board[column2][second_column_highest] = self.l_color
                    self.board[column2][second_column_highest + 1] = self.l_color
                    self.board[column2][second_column_highest + 2] = self.l_color
                    self.board[column1][second_column_highest + 2] = self.l_color
                else:
                    self.board[column1][first_column_highest] = self.l_color
                    self.board[column2][first_column_highest] = self.l_color
                    self.board[column2][first_column_highest - 1] = self.l_color
                    self.board[column2][first_column_highest - 2] = self.l_color

        # J piece
        if piece == 5:
            if rotation == 0:
                if column1 > 7:
                    column1 = 7
                    column2 = 8
                    column3 = 9
                else:
                    column2 = column1 + 1
                    column3 = column1 + 2

                first_column_highest = self.get_highest_point(column1)
                second_column_highest = self.get_highest_point(column2)
                third_column_highest = self.get_highest_point(column3)

                if self.dead:
                    return False

                if not first_column_highest == second_column_highest == third_column_highest:
                    created_gap = True

                if first_column_highest >= second_column_highest and first_column_highest >= third_column_highest:
                    self.board[column1][first_column_highest] = self.j_color
                    self.board[column1][first_column_highest + 1] = self.j_color
                    self.board[column2][first_column_highest] = self.j_color
                    self.board[column3][first_column_highest] = self.j_color

                elif second_column_highest >= first_column_highest and second_column_highest >= third_column_highest:
                    self.board[column1][second_column_highest] = self.j_color
                    self.board[column1][second_column_highest + 1] = self.j_color
                    self.board[column2][second_column_highest] = self.j_color
                    self.board[column3][second_column_highest] = self.j_color
                else:
                    self.board[column1][third_column_highest] = self.j_color
                    self.board[column1][third_column_highest + 1] = self.j_color
                    self.board[column2][third_column_highest] = self.j_color
                    self.board[column3][third_column_highest] = self.j_color

            elif rotation == 1:
                if column1 == 9:
                    column1 = 8
                    column2 = 9
                else:
                    column2 = column1 + 1

                first_column_highest = self.get_highest_point(column1)
                second_column_highest = self.get_highest_point(column2)

                if self.dead:
                    return False

                if not second_column_highest - first_column_highest == 2:
                    created_gap = True

                if first_column_highest >= second_column_highest:
                    self.board[column1][first_column_highest] = self.j_color
                    self.board[column1][first_column_highest + 1] = self.j_color
                    self.board[column1][first_column_highest + 2] = self.j_color
                    self.board[column2][first_column_highest + 2] = self.j_color

                elif first_column_highest + 1 == second_column_highest:
                    self.board[column1][first_column_highest] = self.j_color
                    self.board[column1][first_column_highest + 1] = self.j_color
                    self.board[column1][first_column_highest + 2] = self.j_color
                    self.board[column2][first_column_highest + 2] = self.j_color
                else:
                    self.board[column2][second_column_highest] = self.j_color
                    self.board[column1][second_column_highest] = self.j_color
                    self.board[column1][second_column_highest - 1] = self.j_color
                    self.board[column1][second_column_highest - 2] = self.j_color

            elif rotation == 2:
                if column1 > 7:
                    column1 = 7
                    column2 = 8
                    column3 = 9
                else:
                    column2 = column1 + 1
                    column3 = column1 + 2

                first_column_highest = self.get_highest_point(column1)
                second_column_highest = self.get_highest_point(column2)
                third_column_highest = self.get_highest_point(column3)

                if self.dead:
                    return False

                if not first_column_highest == second_column_highest or not first_column_highest - third_column_highest == 1:
                    created_gap = True

                if third_column_highest >= first_column_highest and third_column_highest >= second_column_highest:
                    self.board[column3][third_column_highest] = self.j_color
                    self.board[column3][third_column_highest + 1] = self.j_color
                    self.board[column2][third_column_highest + 1] = self.j_color
                    self.board[column1][third_column_highest + 1] = self.j_color

                elif second_column_highest >= first_column_highest and second_column_highest >= third_column_highest:
                    self.board[column1][second_column_highest] = self.j_color
                    self.board[column2][second_column_highest] = self.j_color
                    self.board[column3][second_column_highest] = self.j_color
                    self.board[column3][second_column_highest - 1] = self.j_color
                else:
                    self.board[column1][first_column_highest] = self.j_color
                    self.board[column2][first_column_highest] = self.j_color
                    self.board[column3][first_column_highest] = self.j_color
                    self.board[column3][first_column_highest - 1] = self.j_color

            else:
                if column1 == 9:
                    column1 = 8
                    column2 = 9
                else:
                    column2 = column1 + 1

                first_column_highest = self.get_highest_point(column1)
                second_column_highest = self.get_highest_point(column2)

                if self.dead:
                    return False

                if not first_column_highest == second_column_highest:
                    created_gap = True

                if second_column_highest >= first_column_highest:
                    self.board[column2][second_column_highest] = self.j_color
                    self.board[column2][second_column_highest + 1] = self.j_color
                    self.board[column2][second_column_highest + 2] = self.j_color
                    self.board[column1][second_column_highest] = self.j_color
                else:
                    self.board[column2][first_column_highest] = self.j_color
                    self.board[column2][first_column_highest + 1] = self.j_color
                    self.board[column2][first_column_highest + 2] = self.j_color
                    self.board[column1][first_column_highest] = self.j_color

        # T piece
        if piece == 6:
            if rotation == 0:

                if column1 > 7:
                    column1 = 7
                    column2 = 8
                    column3 = 9
                else:
                    column2 = column1 + 1
                    column3 = column1 + 2

                first_column_highest = self.get_highest_point(column1)
                second_column_highest = self.get_highest_point(column2)
                third_column_highest = self.get_highest_point(column3)

                if self.dead:
                    return False

                if not first_column_highest == second_column_highest == third_column_highest:
                    created_gap = True

                if first_column_highest >= second_column_highest and first_column_highest >= third_column_highest:
                    self.board[column1][first_column_highest] = self.t_color
                    self.board[column2][first_column_highest] = self.t_color
                    self.board[column2][first_column_highest + 1] = self.t_color
                    self.board[column3][first_column_highest] = self.t_color

                elif second_column_highest >= first_column_highest and second_column_highest >= third_column_highest:
                    self.board[column1][second_column_highest] = self.t_color
                    self.board[column2][second_column_highest] = self.t_color
                    self.board[column2][second_column_highest + 1] = self.t_color
                    self.board[column3][second_column_highest] = self.t_color
                else:
                    self.board[column1][third_column_highest] = self.t_color
                    self.board[column2][third_column_highest] = self.t_color
                    self.board[column2][third_column_highest + 1] = self.t_color
                    self.board[column3][third_column_highest] = self.t_color

            elif rotation == 1:
                column2 = column1 + 1

                if column1 == 9:
                    column1 = 8
                    column2 = 9

                first_column_highest = self.get_highest_point(column1)
                second_column_highest = self.get_highest_point(column2)

                if self.dead:
                    return False

                if not second_column_highest - first_column_highest == 1:
                    created_gap = True

                if first_column_highest >= second_column_highest:
                    self.board[column1][first_column_highest] = self.t_color
                    self.board[column1][first_column_highest + 1] = self.t_color
                    self.board[column1][first_column_highest + 2] = self.t_color
                    self.board[column2][first_column_highest + 1] = self.t_color
                else:
                    self.board[column1][second_column_highest] = self.t_color
                    self.board[column2][second_column_highest] = self.t_color
                    self.board[column1][second_column_highest + 1] = self.t_color
                    self.board[column1][second_column_highest - 1] = self.t_color

            elif rotation == 2:
                if column1 > 7:
                    column1 = 7
                    column2 = 8
                    column3 = 9
                else:
                    column2 = column1 + 1
                    column3 = column1 + 2

                first_column_highest = self.get_highest_point(column1)
                second_column_highest = self.get_highest_point(column2)
                third_column_highest = self.get_highest_point(column3)

                if self.dead:
                    return False

                if not third_column_highest == first_column_highest or not third_column_highest - second_column_highest == 1:
                    created_gap = True

                if second_column_highest >= first_column_highest and second_column_highest >= third_column_highest:
                    self.board[column1][second_column_highest + 1] = self.t_color
                    self.board[column2][second_column_highest] = self.t_color
                    self.board[column2][second_column_highest + 1] = self.t_color
                    self.board[column3][second_column_highest + 1] = self.t_color

                elif first_column_highest >= second_column_highest and first_column_highest >= third_column_highest:
                    self.board[column1][first_column_highest] = self.t_color
                    self.board[column2][first_column_highest] = self.t_color
                    self.board[column2][first_column_highest - 1] = self.t_color
                    self.board[column3][first_column_highest] = self.t_color
                else:
                    self.board[column1][third_column_highest] = self.t_color
                    self.board[column2][third_column_highest] = self.t_color
                    self.board[column2][third_column_highest - 1] = self.t_color
                    self.board[column3][third_column_highest] = self.t_color

            else:
                if column1 == 0:
                    column1 = 1
                    column2 = 0
                else:
                    column2 = column1 - 1

                first_column_highest = self.get_highest_point(column1)
                second_column_highest = self.get_highest_point(column2)

                if self.dead:
                    return False

                if not first_column_highest - second_column_highest == 1:
                    created_gap = True

                if first_column_highest >= second_column_highest:
                    self.board[column1][first_column_highest] = self.t_color
                    self.board[column1][first_column_highest + 1] = self.t_color
                    self.board[column1][first_column_highest + 2] = self.t_color
                    self.board[column2][first_column_highest + 1] = self.t_color
                else:
                    self.board[column1][second_column_highest] = self.t_color
                    self.board[column2][second_column_highest] = self.t_color
                    self.board[column1][second_column_highest + 1] = self.t_color
                    self.board[column1][second_column_highest - 1] = self.t_color

        the_top = max([first_column_highest, second_column_highest, third_column_highest, fourth_column_highest])
        if the_top > 8:
            if self.pieces_placed > 20:
                self.points += 2
            else:
                self.points += 1
        else:
            if self.pieces_placed > 20:
                self.points += 8
            else:
                self.points += 4

        bumpyness = self.get_bumpyness()
        if bumpyness > 0:
            self.points -= (self.get_bumpyness() * 3)
        else:
            self.points += 3

        if created_gap:
            self.points -= 6

        self.check_lines()
        if not self.training:
            self.show(created_gap)
        return True

    def check_lines(self):
        line_data = np.zeros(20, dtype=np.int32)
        lines = []
        for y in range(20):
            for x in range(10):
                if self.board[x][y] > 0:
                    line_data[y] += 1

        for x, line_count in enumerate(line_data):
            if line_count == 10:
                lines.append(x)

        if len(lines) == 0:
            return

        self.lines_cleared += len(lines)
        if len(lines) == 1:
            self.points += 8

        elif len(lines) == 2:
            self.points += 32

        elif len(lines) == 3:
            self.points += 64

        elif len(lines) == 4:
            self.points += 128

        lines.reverse()
        for line_id in lines:
            sliced_board = self.board[:, line_id:]
            for x in range(10):
                sliced_board[x][0] = 0
            sliced_board = np.roll(sliced_board, -1)
            self.board[:, line_id:] = sliced_board

    def get_game_state(self) -> tuple:
        new_piece = self.get_new_piece()
        return_arr = np.zeros(7)
        return_arr[new_piece] = 1
        return_board = np.zeros(10, dtype=np.float32)

        for i in range(10):
            return_board[i] = self.get_highest_point(i)

        return np.concatenate((return_arr, return_board),
                              axis=None, dtype=np.float32), new_piece

    def reset(self):
        self.board = np.zeros((10, 20), dtype=np.int32)
        self.pieces = [0, 1, 2, 3, 4, 5, 6]
        self.grabbed = -1
        self.dead = False
        self.points = 0
        # random.shuffle(self.pieces)
        self.lines_cleared = 0

    def get_bumpyness(self) -> int:
        height_differences = np.zeros(9, dtype=np.float32)
        last_height = 0

        for i in range(10):
            if i > 0:
                height_differences[i - 1] = abs(last_height - self.get_highest_point(i))
                last_height = self.get_highest_point(i)
            else:
                last_height = self.get_highest_point(i)

        return round(np.average(height_differences))

    def get_highest_point(self, check_column) -> int:
        highest_column_point = 18

        while True:
            if self.board[check_column][highest_column_point] > 0:
                break

            if highest_column_point == 0:
                return 0

            highest_column_point -= 1

        if highest_column_point > 15:
            self.dead = True

        return highest_column_point + 1

    def is_dead(self):
        return self.dead

    def show(self, created_gap):
        formatted_board = np.zeros((20, 10, 3), dtype=np.int32)

        for y in range(20):
            for x in range(10):
                if self.board[x][y] > 0:
                    if self.board[x][y] == 1:
                        formatted_board[y][x] = [255, 255, 0]
                    if self.board[x][y] == 2:
                        formatted_board[y][x] = [255, 153, 0]
                    if self.board[x][y] == 3:
                        formatted_board[y][x] = [255, 0, 0]
                    if self.board[x][y] == 4:
                        formatted_board[y][x] = [0, 204, 0]
                    if self.board[x][y] == 5:
                        formatted_board[y][x] = [0, 255, 255]
                    if self.board[x][y] == 6:
                        formatted_board[y][x] = [255, 0, 255]
                    if self.board[x][y] == 7:
                        formatted_board[y][x] = [157, 0, 255]

        plt.imshow(np.flip(formatted_board, 0))
        plt.title(f"created gap?: {created_gap} and bumpyness: {self.get_bumpyness()}")
        plt.savefig(f"fifth_iteration/testing/{self.pieces_placed}_move.png")


