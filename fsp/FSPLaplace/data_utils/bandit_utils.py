import jax 

import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler



def read_bandit_data(
    bandit_name
):
    """
    Load bandit datasets.
    Code adapted from https://github.com/tensorflow/models/tree/archive/research/deep_contextual_bandits.

    params:
    - bandit_name (str): name of the bandit problem.

    returns:
    - contexts (jnp.array): matrix of n contexts.
    - rewards (jnp.array): matrix of n rewards for the k arms.
    - opt_rewards (jnp.array): vector of n expected optimal rewards.
    - opt_actions (jnp.array): vector of n expected optimal actions.
    """
    if bandit_name == "mushroom":
        data = load_mushroom_data()
    elif bandit_name == "financial":
        data = load_financial_data()
    elif bandit_name == "statlog":
        data = load_statlog_data()
    elif bandit_name == "jester":
        data = load_jester_data()
    else:
        raise ValueError("Bandit name not recognized.")
    
    return data


def load_mushroom_data(
    r_noeat=0, 
    r_eat_safe=5, 
    r_eat_poison_bad=-35, 
    r_eat_poison_good=5, 
    prob_poison_bad=0.5
):
    """
    Load bandit game from Mushroom UCI Dataset.
    We assume r_eat_safe > r_noeat, and r_eat_poison_good > r_eat_poison_bad.

    params:
    - r_noeat (float): reward for not eating a mushroom.
    - r_eat_safe (float): reward for eating a non-poisonous mushroom.
    - r_eat_poison_bad (float): reward for eating a poisonous mushroom if harmed.
    - r_eat_poison_good (float): reward for eating a poisonous mushroom if not harmed.
    - prob_poison_bad (float): probability of being harmed by eating a poisonous mushroom.
    
    returns:
    - contexts (jnp.array): matrix of n contexts.
    - rewards (jnp.array): matrix of n (eat_reward, no_eat_reward) rows.
    - opt_rewards (jnp.array): vector of n expected optimal rewards.
    - opt_actions (jnp.array): vector of n expected optimal actions.
    """
    # Define key 
    key = jax.random.PRNGKey(0)

    # Read data
    data = pd.read_csv("../Data/mushroom.csv", header=None).to_numpy()

    # First two cols of data encode whether mushroom is edible or poisonous
    contexts = data[:,2:]
    num_contexts = contexts.shape[0]

    # Standardize contexts
    scaler = StandardScaler()
    contexts = scaler.fit_transform(contexts)

    # Reward for not eating the mushroom
    no_eat_reward = r_noeat * jnp.ones((num_contexts, 1))
    
    # Reward for eating a mushroom
    r_eat_poison = jax.random.choice(
        key,
        a=jnp.array([r_eat_poison_bad, r_eat_poison_good]),
        p=jnp.array([prob_poison_bad, 1 - prob_poison_bad]),
        shape=(num_contexts,), 
        replace=True
    )
    eat_reward = r_eat_safe * data[:, 0] + r_eat_poison * data[:, 1]
    eat_reward = eat_reward.reshape(num_contexts, 1)

    # Optimal expected reward 
    exp_eat_poison_reward = r_eat_poison_bad * prob_poison_bad
    exp_eat_poison_reward += r_eat_poison_good * (1 - prob_poison_bad)
    opt_exp_reward = r_eat_safe * data[:, 0] 
    opt_exp_reward += max(r_noeat, exp_eat_poison_reward) * data[:, 1]

    # Optimal expected action : no eat = 0 ; eat = 1
    if r_noeat > exp_eat_poison_reward:
        opt_actions = data[:,0] # indicator of edible
    else:
        opt_actions = jnp.ones((num_contexts, 1)) # should always eat

    return (
        contexts, 
        jnp.hstack([no_eat_reward, eat_reward]), 
        opt_exp_reward.reshape(-1, 1), 
        opt_actions.reshape(-1, 1)
    )


def load_financial_data(
    context_dim=21, 
    num_actions=8, 
    sigma=0.1
):
    """
    Load linear bandit game from stock prices dataset.

    params:
    - context_dim (int): context dimension (i.e. vector with the price of each stock).
    - num_actions (int): number of actions (different linear portfolio strategies).
    - sigma (float): additive noise levels for each action.
    
    returns:
    - contexts (jnp.array): matrix of n contexts.
    - rewards (jnp.array): matrix of n rewards for the k arms.
    - opt_rewards (jnp.array): vector of n expected optimal rewards.
    - opt_actions (jnp.array): vector of n expected optimal actions.
    """
    # Define key 
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key, 2)

    # Read data
    contexts = pd.read_csv("../Data/financial.csv",header=None).to_numpy()

    # Get contexts
    contexts = contexts[:, :context_dim]

    # Standardize contexts
    scaler = StandardScaler()
    contexts = scaler.fit_transform(contexts)

    # Sample portfolio coefficients
    betas = jax.random.uniform(key1, minval=-1, maxval=1, shape=(context_dim, num_actions))
    betas /= jnp.linalg.norm(betas, axis=0)

    # Compute portfolio values
    mean_rewards = jnp.dot(contexts, betas)
    noise = mean_rewards + sigma * jax.random.normal(key2, shape=mean_rewards.shape)
    rewards = mean_rewards + noise

    # Optimal expected reward and action
    opt_actions = jnp.argmax(mean_rewards, axis=1)
    opt_rewards = [mean_rewards[i, a] for i, a in enumerate(opt_actions)]

    return (
        contexts, 
        rewards, 
        jnp.array(opt_rewards).reshape(-1, 1), 
        opt_actions.reshape(-1, 1)
    )


def load_jester_data(
    context_dim=32, 
    num_actions=8, 
):
    """
    Load bandit game from (user, joke) dense subset of Jester dataset.
    Consider 9000 samples.

    params:
    - context_dim (int): context dimension (i.e. vector with some ratings from a user).
    - num_actions (int): number of actions (number of joke ratings to predict).

    returns:
    - contexts (jnp.array): matrix of n contexts.
    - rewards (jnp.array): matrix of n rewards for the k arms.
    - opt_rewards (jnp.array): vector of n expected optimal rewards.
    - opt_actions (jnp.array): vector of n expected optimal actions.
    """
    # Read data    
    data = pd.read_csv("../Data/jester.csv", header=None).to_numpy()

    assert context_dim + num_actions == data.shape[1], 'Wrong data dimensions.'

    # Get context and rewards 
    num_contexts = 9000
    contexts = data[:num_contexts,:context_dim].astype('float64')
    rewards = data[:num_contexts,context_dim:].astype('float64')

    # Standardize contexts
    # scaler = StandardScaler()
    # contexts = scaler.fit_transform(contexts)

    # Scale rewards in [0, 1]
    # rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min())

    # Optimal expected reward and action
    opt_actions = jnp.argmax(data[:num_contexts,context_dim:], axis=1)
    opt_rewards = jnp.array(
        [
            data[i, context_dim + a]
            for i, a in enumerate(opt_actions)
        ]
    )

    return (
        contexts, 
        rewards, 
        opt_rewards.reshape(-1, 1), 
        opt_actions.reshape(-1, 1)
    )


def load_statlog_data():
    """
    Load bandit problem dataset based on the UCI statlog data.
    Consider n=9000 contexts and k=7 actions.

    returns:
    - contexts (jnp.array): matrix of n contexts.
    - rewards (jnp.array): matrix of n rewards for the k arms.
    - opt_rewards (jnp.array): vector of n expected optimal rewards.
    - opt_actions (jnp.array): vector of n expected optimal actions.
    """
    data = pd.read_csv("../Data/statlog.csv",header=None).to_numpy()

    num_actions = 7  # some of the actions are very rarely optimal

    # Last column is label, rest are features
    num_contexts = 9000
    contexts = data[:num_contexts,:-1].astype('float64')
    labels = data[:num_contexts,-1].astype(int) - 1  # convert to 0 based index

    # Standardize contexts
    # scaler = StandardScaler()
    # contexts = scaler.fit_transform(contexts)

    # One hot encode labels as rewards
    rewards = np.zeros((num_contexts, num_actions))
    rewards[np.arange(num_contexts), labels] = 1.0

    return (
        contexts, 
        jnp.array(rewards), 
        jnp.ones((num_contexts,1)), 
        labels.reshape(-1, 1)
    )


def plot_bandit_regret(
    cumulative_regrets,
    uniform_cumulative_regrets,
    config
):
    """
    Plot cumulative regret of bandit simulation.

    params:
    - cumulative_regrets (jnp.array): matrix of cumulative regrets across simulations.
    - uniform_cumulative_regrets (jnp.array): matrix of cumulative regrets for uniform policy.
    - config (dict): configuration dictionary.
    """
    # Compute mean and std of cumulative regret across simulations
    mean_cumulative_regret = jnp.mean(cumulative_regrets, axis=0)
    std_cumulative_regret = jnp.std(cumulative_regrets, axis=0)

    # Compute mean and std of cumulative regret for uniform policy
    mean_uniform_cumulative_regret = jnp.mean(uniform_cumulative_regrets, axis=0)
    std_uniform_cumulative_regret = jnp.std(uniform_cumulative_regrets, axis=0)

    x = range(len(mean_cumulative_regret))
    
    # Plot mean of uniform policy regret
    plt.plot(x, mean_uniform_cumulative_regret, color="red", label="Uniform")
    
    # Plot mean 
    plt.plot(mean_cumulative_regret, color="green", label="Thompson sampling")
    
    # Plot std
    plt.fill_between(
        x,
        mean_uniform_cumulative_regret-std_uniform_cumulative_regret,
        mean_uniform_cumulative_regret+std_uniform_cumulative_regret,
        color="red",
        alpha=0.2
    )
    plt.fill_between(
        x,
        mean_cumulative_regret-std_cumulative_regret,
        mean_cumulative_regret+std_cumulative_regret,
        color="green",
        alpha=0.2
    )
    plt.xlabel("Step")
    plt.ylabel("Cumulative regret")
    plt.savefig(f"{config['bandits']['name']}_{config['model']['name']}_regret.pdf")
    plt.show()


