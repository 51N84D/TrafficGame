import torch
from torch import nn
import numpy as np
from torch.optim import Adam
from tqdm import tqdm

"""
Two-step Payoff
              go/go        go/stop        stop/go       stop/stop
          ----------------------------------------------------------
go/go     | -200, -200  |  -99, -101  |  -99, -101  |  2,   -10
go/stop   | -101, -99   |  -101, -101 |  0,   0     |  0,   -10
stop/go   | -101, -99   |  0,    0    |  -101, -101 |  0,   -10
stop/stop | -10,  2     |  -10,  0    |  -10, 0     |  -10, -10
          ----------------------------------------------------------
"""

TWO_STEP_PAYOFF_1 = torch.from_numpy(np.asarray([
    [-200, -99, -99, 2],
    [-101, -101, 0, 0],
    [-101, 0, -101, 0],
    [-5, -5, -5, -5],
])).float()

TWO_STEP_PAYOFF_2 = torch.from_numpy(np.asarray([
    [-200, -101, -101, -5],
    [-99, -101, 0, -5],
    [-99, 0, -101, -5],
    [2, 0, 0, -5],
])).float()


# Define model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden_layers, num_classes):
        super(MLP, self).__init__()
        self.model = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for i in range(n_hidden_layers - 1):
            self.model += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.model += [nn.Linear(hidden_size, num_classes)]
        self.model = nn.Sequential(*self.model)

        self.normalize = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.model(x)
        out = self.normalize(out)
        return out


def sample_z(mean=0, std=1, size=16, batch_size=1):
    return torch.from_numpy(
        np.random.normal(mean, std, size=(batch_size, size))
    ).float()


def train():
    player_1 = MLP(16, 16, 4, 4)
    player_2 = MLP(16, 16, 4, 4)

    opt_1 = Adam(player_1.parameters())
    opt_2 = Adam(player_2.parameters())

    for i in tqdm(range(10000)):
        z = sample_z(size=16, batch_size=1)

        p = player_1(z)
        q = player_2(z)

        loss_1 = -torch.matmul(p, torch.matmul(TWO_STEP_PAYOFF_1, q.T.detach()))
        loss_2 = -torch.matmul(p.detach(), torch.matmul(TWO_STEP_PAYOFF_2, q.T))

        player_1.zero_grad()
        loss_1.backward()
        opt_1.step()

        player_2.zero_grad()
        loss_2.backward()
        opt_2.step()

    print('-----------------------')
    print('p: ', torch.round(p))
    print('payoff 1:', -torch.round(loss_1))
    print('q: ', torch.round(q))
    print('payoff 2: ', -torch.round(loss_2))
    print('-----------------------')


if __name__ == "__main__":
    train()