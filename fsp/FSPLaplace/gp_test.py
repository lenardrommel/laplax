import jax
import copy
import wandb
import torch
import gpytorch

import numpy as np
import jax.scipy as jsp
import jax.numpy as jnp

from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.likelihoods import (
    GaussianLikelihood,
    SoftmaxLikelihood
)


data = np.load("t2m_0_0.npz")
X = data["x"]
y = data["t2m"]

from matplotlib import pyplot as plt    
plt.scatter(X,y)
plt.show()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, y, likelihood, mean_fn, kernel):
        super(ExactGPModel, self).__init__(train_x, y, likelihood)
        self.mean_module = mean_fn
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

kernel = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.MaternKernel(
    nu=1.5,
    active_dims=[0], 
    ard_num_dims=1
)
)

kernel.base_kernel.outputscale = 1
kernel.base_kernel.lengthscale = 1 # torch.FloatTensor([delta_x, delta_y, delta_z])
kernel.outputscale = 10**2

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
    
# Only compatible with Gaussian likelihood
likelihood = GaussianLikelihood()
likelihood.noise =1e-4

mean_fn = gpytorch.means.ZeroMean()

# Define GP model
model = ExactGPModel(
    X, 
    y, 
    likelihood, 
    mean_fn, 
    kernel
)

# Fit model 
model.train()
likelihood.train()

# Define objective
mll = gpytorch.mlls.ExactMarginalLogLikelihood(
    likelihood, 
    model
)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Training loop
optimal_mll, no_improve_count = jnp.inf, 0
for i in range(0):
    optimizer.zero_grad()
    output = model(X)
    loss = -mll(output, y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {i} - Loss: {loss.item():.3f}')
    


# Evaluation mode
model.eval()
likelihood.eval()

# Sample from function distribution
with torch.no_grad():
    f_dist = model(X)
    f_sample = f_dist.sample(sample_shape=torch.Size((10,))).cpu()

f_sample = f_sample.reshape(10, -1)

# Plot
plt.scatter(X.cpu().numpy(), y.cpu().numpy())
plt.plot(X.cpu().numpy(), f_sample.mean(0).cpu().numpy().T, color="red", alpha=0.5)
plt.show()