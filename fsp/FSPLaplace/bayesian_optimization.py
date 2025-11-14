import gc
import os
import jax 
import sys
import time 
import copy
import torch 
import wandb 
import botorch
import traceback

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from botorch.test_functions import *

from datetime import datetime
from gpytorch import distributions as gdists
from botorch.posteriors.gpytorch import GPyTorchPosterior

from models.model import Model
from data_utils.dataloader import DataLoader
from data_utils.dataset import Dataset

from data_utils.bayes_opt_utils import (
    BnnDraw, KnowledgeDistillation, LunarLanderProblem,
    Optics, PDEVar, PestControl, PolyDraw)



def bayesian_optimization(
    config 
):
    """
    params:
    - config (dict): configuration dictionary.
    """
    # Get configuration
    problem_name = config["bayes_opt"]["name"] 
    n_trials = config["bayes_opt"]["n_trials"]
    n_init_points = config["bayes_opt"]["n_init_points"]
    model_id = config["model"]["name"]
    start_trial = config["bayes_opt"]["start_trial"]

    # Initialize wandb
    init_wandb(config)

    # Define random key 
    key = jax.random.PRNGKey(0)
    key, key1 = jax.random.split(key, 2)

    # Create save directory
    save_dir = datetime.now().strftime(f"experiment_results/{problem_name}_{model_id}_%y_%m_%d-%H_%M_%S")
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        jax.config.update("jax_enable_x64", True)
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)

        # get test function
        print("Problem name:", problem_name, "Model name:", model_id)
        test_function = get_test_function(problem_name)
        output_dim = 1
        input_dim = test_function.dim
    
        # Repeat simulation for n_trials
        for trial in range(start_trial, n_trials):
            # Split random key
            key, key1 = jax.random.split(key, 2)
            
            # Display configuration
            print("-" * 20, f"START TRIAL {trial+1} / {n_trials}", "-" * 20)
            if hasattr(test_function, '._optimal_value'):
                print("True minimum:", test_function._optimal_value)

            # Get initial points
            init_x, init_y = initialize_points(test_function, n_init_points, input_dim, output_dim, device, problem_name)

            # Run bayesian optimization
            model_save_dir = "%s/trial_%d/" % (save_dir, trial)
            os.makedirs(model_save_dir)

            print("-" * 20, "running " + model_id, "-" * 20)
            start_time = time.time()
            model = BoTorchModel(config, key1, device)
            best_x, best_y = bayes_opt(
                model, test_function, config, init_x, init_y, device, problem_name, model_save_dir)

            print("\nMax value found was", best_y)
            print("Time(s):", time.time() - start_time)

            del model
            gc.collect()
            jax.clear_caches()
        
        os.rename(save_dir, save_dir + "_done")
        save_dir = save_dir + "_done"
        print("Done!")
    except:
        print(traceback.print_exc())
        os.rename(save_dir, save_dir + "_canceled")
        save_dir = save_dir + "_canceled"
    
    # Plot results
    print("Plotting results...")
    #plot(save_dir, config)



######################### Helper functions ###################################



def bayes_opt(model, test_function, config, init_x, init_y, device, problem_name, save_dir):
    q = int(config["bayes_opt"]["batch_size"])
    output_dim = init_y.shape[-1]
    bounds = test_function.bounds.to(init_x)

    standard_bounds = torch.zeros(2, test_function.dim).to(init_x)
    standard_bounds[1] = 1

    train_x = init_x
    train_y = init_y

    for i in range(config["bayes_opt"]["n_BO_iters"]):
        sys.stdout.flush()
        sys.stderr.flush()
        print("\niteration %d" % i)

        # Fit model on normalized x
        model_start = time.time()
        normalized_x = botorch.utils.transforms.normalize(train_x, bounds).to(train_x)
        model.fit(normalized_x, train_y, q=q)
        model_end = time.time()
        print("fit time", model_end - model_start)
        
        acq_start = time.time()
        sampler = botorch.sampling.stochastic_samplers.StochasticSampler(sample_shape=torch.Size([128]))
        acquisition = botorch.acquisition.qExpectedImprovement(model=model, best_f=train_y.max().item(), sampler=sampler)
        normalized_candidates, acqf_values = botorch.optim.optimize_acqf(
            acquisition, standard_bounds, q=q, num_restarts=2, raw_samples=16, return_best_only=False,
            options={"batch_limit": 1, "maxiter": 10, "with_grad":False})
        candidates = botorch.utils.transforms.unnormalize(normalized_candidates.detach(), bounds=bounds)

        # round candiates
        if problem_name == "pest":
            candidates = torch.floor(candidates)
        # calculate acquisition values after rounding
        normalized_rounded_candidates = botorch.utils.transforms.normalize(candidates, bounds)
        acqf_values = acquisition(normalized_rounded_candidates)
        acq_end = time.time()
        print("acquisition time", acq_end - acq_start)

        best_index = acqf_values.max(dim=0).indices.item()
        # best x is best acquisition value after rounding
        new_x = candidates[best_index].to(train_x)

        del acquisition
        del acqf_values
        del normalized_candidates
        del normalized_rounded_candidates
        gc.collect()
        jax.clear_caches()

        # evaluate new y values and save
        new_y = test_function(new_x)
        # add explicit output dimension
        if output_dim == 1:
            new_y = new_y.unsqueeze(-1)
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        
        print("Max value", train_y.max().item())


    if save_dir is not None:
        torch.save(train_x.cpu(), "%s/train_x.pt" % save_dir)
        torch.save(train_y.cpu(), "%s/train_y.pt" % save_dir)

    max_index = torch.argmax(train_y)
    return train_x[max_index], train_y[max_index]



def initialize_points(test_function, n_init_points, input_dim, output_dim, device, problem_name):
    if n_init_points < 1:
        init_x = torch.ones(1, input_dim, dtype=torch.float64).to(device)
    else:
        bounds = test_function.bounds.to(device, dtype=torch.float64)
        init_x = botorch.utils.sampling.draw_sobol_samples(bounds=bounds, n=n_init_points, q=1).squeeze(-2)
        init_x = init_x.to(device, dtype=torch.float64)
        if problem_name == "pest":
            init_x = torch.floor(init_x)
    init_y = test_function(init_x)

    # add explicit output dimension
    if output_dim == 1:
        init_y = init_y.unsqueeze(-1)

    return init_x, init_y


def get_test_function(test_function):
    test_function = test_function.lower()
    if "ackley" in test_function:
        if test_function == "ackley":
            dim = 2
        else:
            dim = int(test_function.split('_')[-1])
        return Ackley(dim=dim, negate=True)
    elif test_function == "branin":
        return Branin(negate=True)
    elif test_function == "hartmann":
        return Hartmann(negate=True)
    elif test_function == "pde":
        return PDEVar(negate=True)
    elif test_function == "lunar":
        return LunarLanderProblem()
    elif test_function == "pest":
        return PestControl(negate=True)
    elif test_function == "optics":
        return Optics()
    elif test_function == "kd":
        return KnowledgeDistillation()
    elif "bnn" in test_function:
        dim = int(test_function.split('_')[1])
        obj = int(test_function.split('_')[2])
        return BnnDraw(dim, obj, seed=0)
    elif "poly" in test_function:
        dim = int(test_function.split('_')[1])
        return PolyDraw(dim, seed=0)
    else:
        raise NotImplementedError(
            "Test function %s does not exist." % test_function)
    

def jax_array_to_torch_tensor(x):
    if isinstance(x, jnp.ndarray):
        tensor = torch.from_dlpack(jax.dlpack.to_dlpack(x))
    elif isinstance(x, torch.Tensor):
        tensor = x
    else:
        raise Exception("Unknown data type.")
    
    return tensor.cuda() if torch.cuda.is_available() else tensor.cpu()


def torch_tensor_to_jax_array(x):
    return jax.dlpack.from_dlpack(torch.to_dlpack(x.cpu()))


############################## BoTorch Model ##################################

class BoTorchModel(botorch.models.model.Model):

    def __init__(self, config, key, device):
        super().__init__()  
        self.key = key
        self.device = device

        self.config = copy.deepcopy(config)
        self.model_name = self.config["model"]["name"]
        
        # Define model
        self.model = Model(key, self.config)


    def fit(
        self,
        X,
        y, 
        q
    ):
        """
        Fit the model.
        """
        # Keys
        self.key, key = jax.random.split(self.key)

        # Normalize data
        self.X_mean, self.X_std = X.mean(0), X.std(0)
        self.y_mean, self.y_std = y.mean(0), y.std(0)
        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std

        # Convert to jax array
        X = torch_tensor_to_jax_array(X)
        y = torch_tensor_to_jax_array(y)

        # Build dataset
        dataset = Dataset(X, y)
        
        # Build dataloader
        dataloader = DataLoader(key, dataset, q, shuffle=True, replacement=False)

        # Fit model
        self.model = Model(key, self.config)
        self.model.fit(dataloader, dataloader, dataloader)


    def posterior(
        self,
        X, 
        posterior_transform=None,
        **kwargs
    ):
        """
        Return the posterior.
        """
        X = (X - self.X_mean) / self.X_std

        if self.model_name == "FVI":
            post_pred = FVIPosterior(X, self.model, self.y_mean, self.y_std, self.device)
        else:
            if self.model_name == "GP":
                dist = self.model.model.y_distribution_torch(X)
                mean_y = dist.mean * self.y_std + self.y_mean
                cov_y = dist.covariance_matrix * self.y_std**2 
                # Build Gaussian posterior
                dist = gdists.MultivariateNormal(mean_y, covariance_matrix=cov_y)
            else:
                if len(X.shape) < 3:
                    B, D = X.shape
                    Q = 1
                    X = X.reshape(B, Q, D)
                else:
                    B, Q, D = X.shape

                # To numpy array
                if isinstance(X, torch.Tensor):
                    _X = X.cpu().detach().numpy()

                # Get mean and covariance
                ll_var = self.model.model.ll_scale.item()**2
                mean_cov_fn = lambda x: self.model.f_distribution_mean_cov(x, self.key, mc_samples=1)
                mean_y, cov_y = jax.vmap(mean_cov_fn)(_X)
                
                # Reshape
                mean_y, cov_y = mean_y.reshape(B, Q), cov_y.reshape(B, Q, Q)
                
                # Convert to torch tensor
                mean_y = jax_array_to_torch_tensor(mean_y).to(self.device)
                cov_y = jax_array_to_torch_tensor(cov_y).to(self.device) + ll_var * torch.eye(Q).reshape(1, Q, Q).repeat(B, 1, 1).to(X)

                # Rescale
                mean_y = mean_y * self.y_std + self.y_mean
                cov_y = cov_y * self.y_std**2 
                # Build Gaussian posterior
                dist = gdists.MultivariateNormal(mean_y, covariance_matrix=cov_y)
            
            # Return Gaussian posterior
            post_pred = GPyTorchPosterior(dist)

        return post_pred
    
    @property
    def num_outputs(self):
        return 1


class FVIPosterior(botorch.posteriors.posterior.Posterior):

    def __init__(self, X, model, y_mean, y_std, device):
        super().__init__()
        self.model = model
        self.y_mean = y_mean
        self.y_std = y_std
        self._device = device
        self.key = model.model.key

        if isinstance(X, torch.Tensor):
            self.X = X.cpu().detach().numpy()
        else:
            self.X = X

        if len(self.X.shape) < 3:
            self.B, D = self.X.shape
            self.Q = 1
            self.X = self.X.reshape(self.B, self.Q, D)
        else:
            self.B, self.Q, D = self.X.shape

    def rsample(
        self,
        sample_shape,
    ):
        r"""Sample from the posterior."""

        self.key, sub_key = jax.random.split(self.key)
        y_sample = jax.vmap(
            lambda x: self.model.predict_y(x, sub_key, mc_samples=sample_shape[0])
        )(self.X).reshape(sample_shape[0], self.B, self.Q)
        y_sample = jax_array_to_torch_tensor(y_sample).to(self._device) * self.y_std + self.y_mean
        if self.Q == 1:
            y_sample = y_sample.reshape(sample_shape[0], self.B, self.Q, 1)
        else:
            y_sample = y_sample.reshape(sample_shape[0], self.B, self.Q)

        return y_sample

    @property
    def device(self):
        return self._device 

    @property
    def dtype(self):
        r"""The torch dtype of the distribution."""
        return self.X.dtype



def plot(save_dir, config):
    
    dir = save_dir + "/"

    # read arguments from config
    n_trials = config["bayes_opt"]["n_trials"]
    test_function = config["bayes_opt"]["name"]
    init_points = config["bayes_opt"]["n_init_points"]

    plt.figure(figsize=(8, 6))
    for i, model_id in enumerate(["FSPLaplace", "GP", "Laplace"]):
        # store maximum reward
        max_reward = torch.tensor([])
        for t in range(n_trials):
            model_dir = dir + ("trial_%d/" % t) + model_id + "/"
            train_y = torch.load(model_dir + "train_y.pt")

            max_trial_reward = torch.zeros_like(train_y)
            for i in range(len(max_trial_reward)):
                max_trial_reward[i] = train_y[:(i+1)].max()

            max_trial_reward = max_trial_reward[init_points:]
            max_reward = torch.cat((max_reward, max_trial_reward.unsqueeze(0)))

        reward_mean = max_reward.mean(dim=0).reshape(-1)
        reward_std_error = max_reward.std(dim=0, unbiased=False).reshape(-1) / np.sqrt(n_trials)

        xs = range(init_points, len(reward_mean) + init_points)
        plt.plot(xs, reward_mean, label=model_id, linewidth=4)
        plt.fill_between(xs, (reward_mean-reward_std_error), (reward_mean+reward_std_error), alpha=0.1)

    plt.legend()
    plt.title(test_function)
    plt.xlabel("Function Evaluations")
    plt.ylabel("Max Reward")
    plt.tight_layout()
    plt.savefig(dir + test_function + "_plot.pdf", bbox_inches="tight")
    #plt.savefig(dir + test_function + "_plot.png", bbox_inches="tight")





def init_wandb(
    config
):
    """
    """
    # Initialize wandb
    wandb_init = False
    while not wandb_init:
        try:
            wandb.init(
                project="flaplace",
                config=config, 
                name=f'{config["model"]["name"]}_{config["bayes_opt"]["name"]}', 
                #group=f'bayes_opt', 
                mode="offline"
            )
            wandb_init = True
        except:
            time.sleep(10)