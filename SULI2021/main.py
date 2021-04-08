# from random_tools.DataAnalysis import *
from random_tools.DataPrep import *
from models.ensemble import *
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

    dir = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/'

    to_parquet()
    exit()

    greater_than = elmdf.loc[elmdf.index.get_level_values(level='t_to_elm') >= 10]
    less_than = elmdf.loc[elmdf.index.get_level_values(level='t_to_elm') <= 10]
    for elm_no, newdf in greater_than.groupby(level=0):
        print(len(newdf.index))
    for elm_no, newdf in less_than.groupby(level=0):
        print(len(newdf.index))
