import torch.nn as nn
import torch

class ELMRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ELMRNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_vector, hidden):
        combined = torch.cat((input_vector[None, :], hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.sigmoid(output)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)