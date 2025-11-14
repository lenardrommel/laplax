import os
import jax 
import json
import yaml 
import argparse
import matplotlib

import numpy as np

from ood_detection_regression import ood_detection_regression, ood_detection_regression_all
from posterior_comparison import posterior_comparaison_all
from predictive_prior_check import predictive_prior_check
from uci_regression import uci_regression
from toy_data_experiments import toy_data_experiments
from hpo_regression import hpo_regression
from hpo_classification import hpo_classification
from experiments_classification import experiments_classification
from image_classification import image_classification
from era5_interpolation_experiment import era5_interpolation_experiment, hpo_era5_interpolation_experiment
from era5_extrapolation_experiment import era5_extrapolation_experiment, hpo_era5_extrapolation_experiment
from era5_extrapolation_final_experiment import era5_extrapolation_final_experiment
from ocean_modeling import ocean_current_modeling
from bayesian_optimization import bayesian_optimization
from mona_loa_experiments import mona_loa_experiments

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams['font.size'] = 7
matplotlib.rcParams['xtick.labelsize'] = 7
matplotlib.rcParams['ytick.labelsize'] = 7

import matplotlib as mpl
mpl.style.use('default')

os.environ["WANDB__SERVICE_WAIT"] = "300"

if __name__ == "__main__":

    print("Welcome to FSPLaplace!", flush=True)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file', default='sin.yaml', type=str, 
        help="Config file for the model and experiment"
    )
    args = parser.parse_args()
        
    # Load configuration
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
        #config = yaml.safe_load('sin.yaml')

    if config["model"]["name"] == "GP":
        import torch 
        print("Torch default: ", torch.ones(2).device, flush=True)
        jax.config.update('jax_platform_name', 'cpu')
        print("JAX default: ", jax.numpy.ones((2,)).devices(), flush=True)

    jax.config.update("jax_debug_nans", True)
    if config["model"]["name"] == "Laplace":
        jax.config.update("jax_enable_x64", False)
    else:
        jax.config.update("jax_enable_x64", True)

    # Define random key 
    key = jax.random.PRNGKey(0)
    key, key1, key2, key3, key4, key5, key6 = jax.random.split(key, num=7)

    # Fix random seed
    np.random.seed(0)

    # Print configuration
    print(json.dumps(config, sort_keys=True, indent=4))

    # Experiment selection
    if config["experiment"]["name"] == "toy_data_experiments":
        toy_data_experiments(config)
    elif config["experiment"]["name"] == "uci_regression":
        uci_regression(config)
    elif config["experiment"]["name"] == "ood_detection_regression":
        ood_detection_regression(config) #ood_detection_regression_all(config) # ood_detection_regression
    elif config["experiment"]["name"] == "hpo_regression":
        hpo_regression(config)
    elif config["experiment"]["name"] == "posterior_comparison":
        posterior_comparaison_all(config)
    elif config["experiment"]["name"] == "hpo_classification":
        hpo_classification(config)
    elif config["experiment"]["name"] == "experiments_classification":
        experiments_classification(config)
    elif config["experiment"]["name"] == "image_classification":
        image_classification(config)
    elif config["experiment"]["name"] == "predictive_prior_check":
        predictive_prior_check(config)
    elif config["experiment"]["name"] == "era5_interpolation":
        era5_interpolation_experiment(config)
    elif config["experiment"]["name"] == "hpo_era5_interpolation":
        hpo_era5_interpolation_experiment(config)
    elif config["experiment"]["name"] == "era5_extrapolation":
        era5_extrapolation_experiment(config)
    elif config["experiment"]["name"] == "era5_extrapolation_final_experiment":
        era5_extrapolation_final_experiment(config)
    elif config["experiment"]["name"] == "hpo_era5_extrapolation":
        hpo_era5_extrapolation_experiment(config)
    elif config["experiment"]["name"] == "ocean_current_modeling":
        ocean_current_modeling(config)
    elif config["experiment"]["name"] == "bayesian_optimization":
        bayesian_optimization(config)
    elif config["experiment"]["name"] == "mona_loa_experiments":
        mona_loa_experiments(config)
    else:
        raise NotImplementedError()