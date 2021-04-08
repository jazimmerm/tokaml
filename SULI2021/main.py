# from random_tools.DataAnalysis import *
from random_tools.DataPrep import *
from models.skl_linear import *
from random_tools.tools import *
import models.torch_linear as tl

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

    inputDim = 4  # takes variable 'x'
    outputDim = 1  # takes variable 'y'
    hiddenDim = 100
    numLayers = 1
    learningRate = 0.0001
    epochs = 2
    batch_size = 64

    file = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/all.pq'

    model = tl.makeModel(file, inputDim, outputDim, hiddenDim, numLayers, batch_size, epochs, learningRate)

    lstm_model, loss_all = model.train_lstm()
    model.test(lstm_model)
    exit()

    greater_than = elmdf.loc[elmdf.index.get_level_values(level='t_to_elm') >= 10]
    less_than = elmdf.loc[elmdf.index.get_level_values(level='t_to_elm') <= 10]
    for elm_no, newdf in greater_than.groupby(level=0):
        print(len(newdf.index))
    for elm_no, newdf in less_than.groupby(level=0):
        print(len(newdf.index))
