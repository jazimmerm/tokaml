import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import models
from tqdm import tqdm

tb = SummaryWriter()

n_samples = 100000

test_spectrogram = torch.sin(torch.linspace(0, n_samples, n_samples))[:,None]
test_events = torch.where(torch.sin(torch.linspace(0, n_samples, n_samples)) > 0.60, 1, 0)
samples = zip(torch.split(test_spectrogram, 10), torch.split(test_events, 10))
net = models.ELMRNN(
    input_size=1,
    hidden_size=10,
    output_size=1
    )
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.05)

def train(net, sample, epoch):

    hidden = net.initHidden()

    net.zero_grad()

    for i in range(sample[0].shape[0]):
        output, hidden = net(sample[0][i,:], hidden)

    optimizer.zero_grad()
    loss = criterion(output, sample[1][-1, None, None].to(torch.float))
    loss.backward()
    optimizer.step()

    tb.add_scalar('loss', loss, epoch)
    tb.add_histogram('i2h.weight', net.i2h.weight, epoch)

    i += 1

    return output, loss.item()

i = 0
for s in tqdm(samples):
    train(net, s, i)
    i = i + 1

tb.flush()
tb.close()