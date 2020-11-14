import os
import sys
import matrixgame.game as gm
import matplotlib.pyplot as plt


class Command():


    def run(self):
        path = os.path.join('.','matrixgame','matrices','matrix_complete.txt')
        a = gm.read_matrix(path)
        p,q = gm.nash_equilibrium(a)

        try:
            i = 0   
            x = [i+1 for i in range(len(p))]
            i = 0
            y1 = [0 for i in range(len(p))]
        except TypeError:
            if p == q == -1:
                sys.exit(1)
            x = p + 1
            p = q + 1
            y1 = 0

        plt.vlines(x, ymin=y1, ymax=p, color='blue')
        plt.scatter(x, p, s=10, color='blue')
        plt.show()

