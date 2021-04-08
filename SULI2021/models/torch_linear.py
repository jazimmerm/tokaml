import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pandas as pd
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


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, hiddenSize):
        super(linearRegression, self).__init__()
        self.dense_h1 = nn.Linear(in_features=inputSize, out_features=hiddenSize)
        self.sigmoid_h1 = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        self.dense_out = torch.nn.Linear(in_features=hiddenSize, out_features=outputSize)


    def forward(self, X):

        out = self.sigmoid_h1(self.dense_h1(X))
        out = self.dropout(out)
        out = self.dense_out(out)
        return out


if __name__ == '__main__':

    inputDim = 4  # takes variable 'x'
    outputDim = 1  # takes variable 'y'
    hiddenDim = 300
    learningRate = 0.01
    epochs = 10
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data
    data = pd.read_parquet('/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/all.pq', engine='pyarrow')
    X = data[['Peak_Amp', 'Peak_Freq', 'width', 't_since_elm']]
    Y = data['t_to_elm']

    # Split the data
    train, test = train_test_split(list(range(X.shape[0])), test_size=.3)
    ds = PrepareData(X.to_numpy(), y=Y.to_numpy(), scale_X=True)
    train_set = DataLoader(ds, batch_size=batch_size,
                           sampler=SubsetRandomSampler(train))
    test_set = DataLoader(ds, batch_size=batch_size,
                          sampler=SubsetRandomSampler(test))

    model = linearRegression(inputDim, outputDim, hiddenDim).to(device)

    cost_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    loss_all = []
    for epoch in range(epochs):
        batch_losses = []
        for i, (X_train, Y_train) in enumerate(train_set):
            # Converting inputs and labels to Variable
            inputs = Variable(X_train).float().to(device)
            target = Variable(Y_train).float().to(device)

            # ==========Forward pass===============
            # get output from the model, given the inputs
            outputs = model(inputs)

            # get loss for the predicted output
            loss = cost_func(outputs, target.unsqueeze(1))
            batch_losses.append(loss.item())
            loss_all.append(loss.item())

            # ==========backward pass==============
            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, don't want to acummulate gradients
            optimizer.zero_grad()
            # get gradients w.r.t to parameters
            loss.backward()
            # update parameters
            optimizer.step()

        mbl = np.mean(np.sqrt(batch_losses)).round(3)
        print(f'epoch {epoch}/{epochs}, mean square batch loss {mbl}')

    print(model.training)
    model.eval()
    print(model.training)

    test_batch_losses = []
    for X_test, y_test in test_set:
        X_test = Variable(X_test).float().to(device)
        y_test = Variable(y_test).float().to(device)

        # apply model
        y_pred = model(X_test)
        test_loss = cost_func(y_pred, y_test.unsqueeze(1))

        test_batch_losses.append(test_loss.item())

    print(f'Mean square test batch loss: {np.mean(np.sqrt(test_batch_losses))}')



    # plt.clf()
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

    '''
        with torch.no_grad():  # we don't need gradients in the testing phase
            if torch.cuda.is_available():
                predicted = model(Variable(torch.from_numpy(X_test).cuda())).cpu().data.numpy()
            else:
                predicted = model(Variable(torch.from_numpy(X_test))).data.numpy()
            print(predicted)
    '''