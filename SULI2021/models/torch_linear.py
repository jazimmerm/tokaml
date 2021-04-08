import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class PrepareData(Dataset):

    def __init__(self, X, y, scale_X=True):
        if not torch.is_tensor(X):
            if scale_X:
                X = StandardScaler().fit_transform(X)
                self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(torch.nn.Module):
    def __init__(self, inputSize, outputSize, hiddenSize, numLayers):
        super(LSTMModel, self).__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers

        self.LSTM = nn.LSTM(input_size=self.inputSize, hidden_size=self.hiddenSize, num_layers=self.numLayers)
        self.dense_h1 = nn.Linear(in_features=self.hiddenSize, out_features=self.outputSize)

    def forward(self, x):
        # initialize hidden layer to zero
        h_0 = torch.zeros(self.numLayers, x.size()[0], self.hiddenSize, requires_grad=True)
        # initialize cell
        c_0 = torch.zeros(self.numLayers, x.size()[0], self.hiddenSize, requires_grad=True)

        out, (h_n, c_n) = self.LSTM(x, (h_0.detach(), c_0.detach()))
        out = self.fc(out[:, -1, :])

        return out


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, hiddenSize):
        super(linearRegression, self).__init__()
        self.dense_h1 = nn.Linear(in_features=inputSize, out_features=hiddenSize)
        self.dense_h2 = nn.Linear(in_features=hiddenSize, out_features=hiddenSize // 2)
        self.dense_out = torch.nn.Linear(in_features=hiddenSize // 2, out_features=outputSize)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, X):
        out = self.dense_h1(X)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.dense_h2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.dense_out(out)
        return out


class makeModel():
    def __init__(self, file, inputDim, outputDim, hiddenDim, numLayers, batch_size, epochs, learningRate, ):

        self.batch_size = batch_size
        self.epochs = epochs
        self.learningRate = learningRate
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.hiddenDim = hiddenDim
        self.numLayers = numLayers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_set, self.test_set = self.get_load_data(file)

    def get_load_data(self, file):
        data = pd.read_parquet(file, engine='pyarrow')
        X = data[['Peak_Amp', 'Peak_Freq', 'width', 't_since_elm']]
        Y = data['t_to_elm']

        # Split the data
        train, test = train_test_split(list(range(X.shape[0])), test_size=.3)
        ds = PrepareData(X.to_numpy(), y=Y.to_numpy(), scale_X=True)
        train_set = DataLoader(ds, batch_size=self.batch_size,
                               sampler=SubsetRandomSampler(train))
        test_set = DataLoader(ds, batch_size=self.batch_size,
                              sampler=SubsetRandomSampler(test))

        return train_set, test_set

    def train_linear(self):
        lin_model = linearRegression(self.inputDim, self.outputDim, self.hiddenDim).to(self.device)

        cost_func = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(lin_model.parameters(), lr=self.learningRate, momentum=0.5)

        loss_all = []
        for epoch in range(self.epochs):
            batch_losses = []
            for i, (X_train, Y_train) in enumerate(self.train_set):
                # Converting inputs and labels to Variable
                inputs = Variable(X_train).float().to(self.device)
                target = Variable(Y_train).float().to(self.device)

                # ==========Forward pass===============
                # get output from the model, given the inputs
                outputs = lin_model(inputs)

                # get loss for the predicted output
                loss = cost_func(outputs, target.unsqueeze(1))
                batch_losses.append(loss.item())

                # ==========backward pass==============
                # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, don't want to acummulate gradients
                optimizer.zero_grad()
                # get gradients w.r.t to parameters
                loss.backward()
                # update parameters
                optimizer.step()

            mbl = np.mean(np.sqrt(batch_losses)).round(3)
            loss_all.append(mbl)
            print(f'epoch {epoch + 1}/{self.epochs}, root mean square batch loss {mbl}')

        return lin_model, loss_all

    def train_lstm(self):
        lstm_model = LSTMModel(self.inputDim, self.outputDim, self.hiddenDim, self.numLayers).to(self.device)

        cost_func = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(lstm_model.parameters(), lr=self.learningRate, momentum=0.5)

        loss_all = []
        for epoch in range(self.epochs):
            batch_losses = []
            for i, (X_train, Y_train) in enumerate(self.train_set):
                # Converting inputs and labels to Variable
                inputs = Variable(X_train).float().to(self.device)
                target = Variable(Y_train).float().to(self.device)

                # ==========Forward pass===============
                # get output from the model, given the inputs
                outputs = lstm_model(inputs)

                # get loss for the predicted output
                loss = cost_func(outputs, target.unsqueeze(1))
                batch_losses.append(loss.item())

                # ==========backward pass==============
                # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, don't want to acummulate gradients
                optimizer.zero_grad()
                # get gradients w.r.t to parameters
                loss.backward()
                # update parameters
                optimizer.step()

            mbl = np.mean(np.sqrt(batch_losses)).round(3)
            loss_all.append(mbl)
            print(f'epoch {epoch + 1}/{self.epochs}, root mean square batch loss {mbl}')

        return lstm_model, loss_all

    def test(self, model: object):
        cost_func = torch.nn.MSELoss()

        model.eval()
        print(model.training)

        test_batch_losses = []
        with torch.no_grad():
            for X_test, y_test in self.test_set:
                X_test = Variable(X_test).float().to(self.device)
                y_test = Variable(y_test).float().to(self.device)

                # apply model
                y_pred = model(X_test)
                test_loss = cost_func(y_pred, y_test.unsqueeze(1))

                test_batch_losses.append(test_loss.item())

            print(f'Mean square test batch loss: {np.mean(np.sqrt(test_batch_losses))}')


if __name__ == '__main__':
    inputDim = 4  # takes variable 'x'
    outputDim = 1  # takes variable 'y'
    hiddenDim = 300
    numLayers = 1
    learningRate = 0.0001
    epochs = 2
    batch_size = 64

    train_set, test_set = get_load_data('/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/all.pq')

    model, model_losses = train_linear(train_set, inputDim, outputDim, hiddenDim)
    test(model, test_set)

    plt.clf()
    plt.plot(range(epochs), model_losses)
    plt.title('Learning Curve for 2-Layer Neural Network')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square-Root Loss')
    plt.show()
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].set_title('Amplitude')
    # axs[0, 0].scatter(X_test['Peak_Amp'].tolist(), y_test, c='black')
    # axs[0, 0].plot(X_test['Peak_Amp'].tolist(), y_pred, c='blue')
    # axs[0, 0].set_xlabel('Amplitude')
    # axs[0, 0].set_ylabel('Time to ELM')
    #
    # axs[0, 1].set_title('Frequency')
    # axs[0, 1].scatter(X['Peak_Freq'].tolist(), Y, c='black')
    # axs[0, 1].plot(X_test['Peak_Freq'].tolist(), y_pred, c='blue')
    # axs[0, 1].set_xlabel('Frequency')
    # axs[0, 1].set_ylabel('Time to ELM')
    #
    # axs[1, 1].set_title('Width')
    # axs[1, 1].scatter(X['width'].tolist(), Y, c='black')
    # axs[1, 1].plot(X_test['width'].tolist(), y_pred, c='blue')
    # axs[1, 1].set_xlabel('width')
    # axs[1, 1].set_ylabel('Time to ELM')
    #
    # axs[1, 0].set_title('Time since Last ELM')
    # axs[1, 0].scatter(X['t_since_elm'].tolist(), Y, c='black')
    # axs[1, 0].plot(X_test['t_since_elm'].tolist(), y_pred, c='blue')
    # axs[1, 0].set_xlabel('t_since_elm')
    # axs[1, 0].set_ylabel('Time to ELM')
    #
    # plt.show()
