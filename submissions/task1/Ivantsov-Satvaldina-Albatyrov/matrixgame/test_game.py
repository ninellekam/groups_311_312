import os
import sys
import unittest
import numpy as np
import matrixgame.game as gm


class TestMatrixgame(unittest.TestCase):


    def setUp(self):
        pass


    def test_nash_equilibrium_0(self):
        path = os.path.join('.','matrixgame','test_matrices','test_matrix_0.txt')
        test_matrix = gm.read_matrix(path)
        test_p = np.round(np.array([2, 3, 2]), decimals=5)
        test_q = np.round(np.array([1, 0, 2]), decimals=5)
        p, q = gm.nash_equilibrium(test_matrix)
        p = np.round(p, decimals=5)
        q = np.round(q, decimals=5)
        self.assertEqual(p.tolist(), test_p.tolist())
        self.assertEqual(q.tolist(), test_q.tolist())

    def test_nash_equilibrium_1(self):
        path = os.path.join('.','matrixgame','test_matrices','test_matrix_1.txt')
        test_matrix = gm.read_matrix(path)
        test_p = np.round(np.array([1/8, 25/52, 19/52, 3/104]), decimals=5)
        test_q = np.round(np.array([1/8, 37/104, 23/52, 1/13]), decimals=5)
        p, q = gm.nash_equilibrium(test_matrix)
        p = np.round(p, decimals=5)
        q = np.round(q, decimals=5)
        self.assertEqual(p.tolist(), test_p.tolist())
        self.assertEqual(q.tolist(), test_q.tolist())

    def test_nash_equilibrium_2(self):
        path = os.path.join('.','matrixgame','test_matrices','test_matrix_2.txt')
        test_matrix = gm.read_matrix(path)
        test_p = np.round(np.array([1/2, 1/2]), decimals=5)
        test_q = np.round(np.array([1/2, 1/2]), decimals=5)
        p, q = gm.nash_equilibrium(test_matrix)
        p = np.round(p, decimals=5)
        q = np.round(q, decimals=5)
        self.assertEqual(p.tolist(), test_p.tolist())
        self.assertEqual(q.tolist(), test_q.tolist())

if __name__ == '__main__':
    unittest.main()

