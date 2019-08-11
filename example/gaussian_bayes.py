import copy

import numpy as np
from itertools import count
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal


class MNormal(nn.Module):
    def __init__(self):
        super(MNormal, self).__init__()
        self.location = torch.nn.Parameter(torch.from_numpy(np.array([0, 0])).float())

    def output(self):
        return self.location

    def prob(self):
        return MultivariateNormal(
            loc=self.location,
            covariance_matrix=torch.nn.Parameter(100*torch.eye(n=2)),
        )


# model
model = MNormal()
model.cpu()

# prior
prior = MNormal()
prior.cpu()

data = torch.from_numpy(
    np.random.multivariate_normal(
        size=200,
        mean=np.array([100, 55]),
        cov=np.array([[100, 0], [0, 300]])
    )
).float()

n_update = 10
n_param_sample = 50
for iteration in count(1):
    # Posterior optimization (approximated Bayes update)
    posterior = copy.deepcopy(prior)
    optimizer = optim.Adam(posterior.parameters(), lr=0.5)
    for n in range(n_update):
        optimizer.zero_grad()
        losses = []
        for i in range(n_param_sample):
            loc = posterior.prob().sample().detach()  # sample a model param
            model.location = torch.nn.Parameter(loc)  # set sampled model parameter into model
            r = (-model.prob().log_prob(data).sum() + posterior.prob().log_prob(loc) - prior.prob().log_prob(loc)).detach()
            losses.append(r * posterior.prob().log_prob(loc))
        loss = torch.stack(losses).sum()
        loss.backward()
        optimizer.step()

    # Update Prior
    prior = copy.deepcopy(posterior)

    print(f"iteration {iteration}")
    print(prior.location)
    loc = prior.output()
    plt.clf()
    plt.scatter(x=data[:, 0], y=data[:, 1])
    plt.scatter(x=[float(loc[0])], y=[float(loc[1])], c="r")
    plt.pause(0.01)
