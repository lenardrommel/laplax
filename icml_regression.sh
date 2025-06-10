#!/bin/bash

python submit/submit.py \
  --mode local \
  --script regression \
  --curv_type full diagonal lanczos \
  --calibration_method grid_search gradient_descent \
  --calibration_objective nll marginal_log_likelihood \
  --pushforward_type linear nonlinear \
  --last_layer_only False \
  --wandb True

python submit/submit.py --mode local --script regression \
  --task evaluate \
  --curv_type full \
  --calibration_method grid_search gradient_descent \
  --calibration_objective nll marginal_log_likelihood \
  --pushforward_type linear nonlinear \
  --last_layer_only True \
  --wandb True
