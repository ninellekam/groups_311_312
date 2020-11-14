import os
import sys
import matrixgame.game as gm
import matplotlib.pyplot as plt


class Command():


    def run(self):
        path = os.path.join('.','matrixgame','matrices','matrix_equilibrium.txt')
        a = gm.read_matrix(path)
        p,q = gm.nash_equilibrium(a)

        eq_i = -1
        eq_j = -1
        row_len = len(a[0])
        col_len = len(a)
        for i in range(row_len):
            if all(p == a[i]):
                eq_i = i
                break
        for j in range(col_len):
            if all(q == a[:,j]):
                eq_j = j
                break
                
        x = eq_i + 1
        y = eq_j + 1
        y1 = 0

        plt.vlines(x, ymin=y1, ymax=y, color='blue')
        plt.scatter(x, y, s=10, color='blue')
        plt.show()



