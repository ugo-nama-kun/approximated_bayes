import numpy as np
from itertools import count
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal


class MultNormal(nn.Module):
    def __init__(self):
        super(MultNormal, self).__init__()
        self.loc = torch.nn.Parameter(torch.randn(size=[2]))
        self.cov = torch.nn.Parameter(torch.eye(n=2))

    def output(self):
        return self.loc, self.cov


model = MultNormal()
model.cpu()
optimizer = optim.Adam(model.parameters(), lr=0.1)

data = torch.from_numpy(
    np.random.multivariate_normal(
        size=200,
        mean=np.array([2, 5]),
        cov=np.array([[1, 0], [0, 3]])
    )
).float()

n_update = 10
for iteration in count(1):
    optimizer.zero_grad()
    loc, cov = model.output()
    prob = MultivariateNormal(
        loc=loc,
        covariance_matrix=cov
    )
    losses = []
    for d in data:
        losses.append(prob.log_prob(d))
    loss = -torch.stack(losses).sum()
    loss.backward()
    optimizer.step()

    print(f"iteration {iteration}")
    loc, _ = model.output()
    print(loc)
    plt.clf()
    plt.scatter(x=data[:, 0], y=data[:, 1])
    plt.scatter(x=[float(loc[0])], y=[float(loc[1])], c="r")
    plt.pause(0.01)
