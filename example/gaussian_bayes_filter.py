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
        self.location = torch.nn.Parameter(torch.from_numpy(np.array([100, 0])).float())

    def output(self):
        return self.location

    def prob(self):
        return MultivariateNormal(
            loc=self.location,
            covariance_matrix=torch.nn.Parameter(100 * torch.eye(n=2)),
        )


# model
model = MNormal()
model.cpu()

# prior
prior = MNormal()
prior.cpu()

# trajectory
radius = 100
t = np.linspace(0, 2*np.pi, 50)
x_traj = radius*np.cos(t)
y_traj = radius*np.sin(t)

n_update = 50
n_param_sample = 5
for iteration in count(0):
    # Observation
    theta = 2*np.pi * iteration/50.0
    data = torch.from_numpy(
        np.random.multivariate_normal(
            size=10,
            mean=np.array([radius*np.cos(theta), radius*np.sin(theta)]),
            cov=np.array([[50, 0], [0, 50]])
        )
    ).float()

    # Posterior optimization (approximated Bayes update)
    posterior = copy.deepcopy(prior)
    optimizer = optim.Adam(posterior.parameters(), lr=1)
    for n in range(n_update):
        optimizer.zero_grad()
        losses = []
        for i in range(n_param_sample):
            loc = posterior.prob().sample().detach()  # sample a model param
            model.location = torch.nn.Parameter(loc)  # set sampled model parameter into model
            # Get a score (ELBO)
            r = (-model.prob().log_prob(data).sum() + posterior.prob().log_prob(loc) - prior.prob().log_prob(
                loc)).detach()
            # print(r)
            losses.append(r * posterior.prob().log_prob(loc))
        loss = torch.stack(losses).sum()
        loss.backward()
        optimizer.step()

    # Update prior
    prior = copy.deepcopy(posterior)

    print(f"iteration {iteration}")
    print(prior.location)
    loc = prior.output()
    plt.clf()
    plt.scatter(x=x_traj, y=y_traj, c="k", marker="+")
    plt.scatter(x=data[:, 0], y=data[:, 1])
    plt.scatter(x=[float(loc[0])], y=[float(loc[1])], c="r")
    plt.xlim([-1.5*radius, 1.5*radius])
    plt.ylim([-1.5*radius, 1.5*radius])
    plt.pause(0.01)
