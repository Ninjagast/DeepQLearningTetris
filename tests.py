import unittest
import random

import numpy as np


class MyTestCase(unittest.TestCase):

    def test_get_lines_ids(self):
        tetris_board = np.zeros((10, 20), dtype=np.int32)  # empty tetris board

        tetris_board[:, 1] = 1  # we fill the second row from the bottom with pieces
        tetris_board[:, 5] = 1  # we fill the sixth row from the bottom with pieces


        line_data = np.zeros(20, dtype=np.int32)
        lines = []
        # we check through the array the opposite way since the array is flipped on its side
        for y in range(20):
            for x in range(10):
                if tetris_board[x][y] > 0:
                    line_data[y] += 1

        for x, line_count in enumerate(line_data):
            if line_count == 10:
                lines.append(x)  # log the line position if it has 10 cells filled

        self.assertEqual([1, 5], lines)  # did we find the two line id's?

    def test_line_removal(self):
        tetris_board = np.zeros((10, 20), dtype=np.int32)  # empty tetris board
        expected_res_board = np.zeros((10, 20), dtype=np.int32)

        expected_res_board[4][4] = 1
        expected_res_board[8][1] = 1

        tetris_board[:, 1] = 1  # we fill the second row from the bottom with pieces
        tetris_board[:, 5] = 1  # we fill the sixth row from the bottom with pieces
        tetris_board[4][6] = 1  # we fill a cell to check the result array
        tetris_board[8][2] = 1  # we fill a cell to check the result array
        line_ids = [1, 5]  # the ids of the rows we filled


        # we reverse the order of the rows since we want to check them from the top down
        line_ids.reverse()
        for line_id in line_ids:  # we go through every filled line
            # we slice the main board from the line we want to clear to the top
            sliced_board = tetris_board[:, line_id:]
            for x in range(10):
                sliced_board[x][0] = 0  # we clear the line
            sliced_board = np.roll(sliced_board, -1)  # we roll the bottom to the top
            tetris_board[:, line_id:] = sliced_board  # we return this slice to the entire board

        # check if the result array and expected array are the same
        np.testing.assert_array_equal(expected_res_board, tetris_board)

    # @unittest.skip
    def test_piece_shuffle(self):
        og_pieces = [0, 1, 2, 3, 4, 5, 6]
        pieces = [0, 1, 2, 3, 4, 5, 6]
        random.shuffle(pieces)

        # small chance that this is going to fail since the randomizer might output the input
        self.assertNotEqual(og_pieces, pieces)

    def test_get_highest_point_from_column(self):
        highest_column_point = 18
        column = 1
        tetris_board = np.zeros((10, 20), dtype=np.int32)  # empty tetris board
        tetris_board[:, 1] = 1  # we fill the second row from the bottom with pieces

        while True:
            if tetris_board[column][highest_column_point] > 0:
                break

            if highest_column_point == 0:
                break

            highest_column_point -= 1

        self.assertEqual(highest_column_point, 1)


if __name__ == '__main__':
    unittest.main()
