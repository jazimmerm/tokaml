# from random_tools.DataAnalysis import *
from random_tools.DataPrep import *
from random_tools.tools import *

if __name__ == '__main__':

    '''
        Shots: 
         [174828.]
         [174829.]
         [174830.]
         [174833.]
         [174860.]
         [174870.]
    '''

    sh = DataPrep(174828)
    plot_split(sh)
    # sh.split()