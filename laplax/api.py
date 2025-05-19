"""Functional API around Laplax's Laplace approximation stack.

This module provides a high-level interface for Laplace approximation in neural networks,
including curvature estimation, hyperparameter calibration, and predictive evaluation.

Public entry points
-------------------
* :func:`laplace`     - fit curvature / posterior fn
* :func:`calibration` - tune prior precision (or similar args)
* :func:`evaluation`  - evaluate performance of calibrated model
"""

from enum import StrEnum
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from loguru import logger

# Laplax imports
from laplax.curv.cov import Posterior, estimate_curvature, set_posterior_fn
from laplax.curv.ggn import create_ggn_mv_without_data
from laplax.enums import (
    CalibrationMethod,
    CalibrationObjective,
    CurvApprox,
    DefaultMetrics,
    LossFn,
    Predictive,
    Pushforward,
)
from laplax.eval import (
    evaluate_for_given_prior_arguments,
    marginal_log_likelihood,
)
from laplax.eval.calibrate import optimize_prior_prec
from laplax.eval.metrics import (
    DEFAULT_REGRESSION_METRICS,
    chi_squared_zero,
    correctness,
    expected_calibration_error,
    nll_gaussian,
)
from laplax.eval.pushforward import (
    # linear
    lin_mc_pred_act,
    lin_pred_mean,
    lin_pred_std,
    lin_setup,
    lin_special_pred_act,
    # non-linear
    nonlin_mc_pred_act,
    nonlin_pred_mean,
    nonlin_pred_std,
    nonlin_setup,
    # general
    set_lin_pushforward,
    set_nonlin_pushforward,
)
from laplax.eval.utils import apply_fns, evaluate_metrics_on_dataset
from laplax.types import (
    Any,
    Array,
    Callable,
    Data,
    Float,
    InputArray,
    Int,
    Iterable,
    KeyType,
    ModelFn,
    Params,
    PriorArguments,
    PyTree,
)
from laplax.util.loader import (
    DataLoaderMV,
    identity,
    input_target_split,
    reduce_add,
)

DEFAULT_KEY = jax.random.key(0)
EMPTY_DICT = {}

# Constants
_SPECIAL_PREDICTIVES = {
    Predictive.LAPLACE_BRIDGE,
    Predictive.MEAN_FIELD_0,
    Predictive.MEAN_FIELD_1,
    Predictive.MEAN_FIELD_2,
}

calibration_options: dict[CalibrationMethod | str, Callable] = {
    CalibrationMethod.GRID_SEARCH: optimize_prior_prec,
}

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------


def _check_if_none(*args: Any) -> bool:
    """Check if any of the provided arguments are None.

    Parameters
    ----------
    *args : Any
        Arguments to check for None values.

    Returns:
    -------
    bool
        True if any argument is None, False otherwise.
    """
    return any(x is None for x in args)


def _validate_and_get_transform(batch: Data | Any) -> Callable[[Any], Data]:
    """Return the transform that converts a single batch to (inputs, targets).

    Parameters
    ----------
    batch : Data | Any
        Batch to validate and get transform for.

    Returns:
    -------
    Callable[[Any], Data]
        Function that transforms a batch into (inputs, targets) format.

    Raises:
    ------
    ValueError
        If batch is not a tuple/dict or misses required keys.
    """
    if isinstance(batch, (tuple, list)):
        if len(batch) != 2:
            msg = "Tuple batches must be `(input, target)` – received len != 2."
            raise ValueError(msg)
        return input_target_split

    if isinstance(batch, dict):
        if "input" not in batch or "target" not in batch:
            msg = "Dict batches must contain keys `'input'` & `'target'`."
            raise ValueError(msg)
        return identity

    msg = f"Unsupported batch type: {type(batch)}. Expect tuple or mapping."
    raise ValueError(msg)


def _maybe_wrap_loader_or_batch(
    mv_fn: Callable[..., Params],
    data: Data | Iterable,
    *,
    loader_kwargs: dict = EMPTY_DICT,
) -> Callable[..., Params]:
    """Wrap matrix-vector function with data loader if data is iterable.

    Parameters
    ----------
    mv_fn : Callable[..., Params]
        Matrix-vector product function to wrap.
    data : Data | Iterable
        Data to use with the function.
    loader_kwargs : dict, optional
        Additional arguments for the data loader.

    Returns:
    -------
    Callable[..., Params]
        Wrapped matrix-vector function.
    """
    transform = _validate_and_get_transform(
        next(iter(data)) if isinstance(data, Iterable)
        and not isinstance(data, (tuple, dict)) else data
    )

    if isinstance(data, (tuple, dict)):
        logger.debug("Using *single batch* curvature evaluation.")
        return partial(mv_fn, data=transform(data))

    logger.debug("Wrapping curvature with streaming DataLoaderMV.")
    return DataLoaderMV(
        mv=mv_fn,
        loader=data,
        transform=transform,
        reduce=loader_kwargs.pop("reduce", reduce_add),
        **loader_kwargs,
    )


def _convert_to_enum(
    enum_cls: type[StrEnum],
    value: StrEnum | str,
    *,
    str_default: bool = False,
) -> StrEnum:
    """Convert string to enum, pass through if already enum.

    Parameters
    ----------
    enum_cls : StrEnum
        Enum class to convert to.
    value : StrEnum | str
        Value to convert.
    str_default : bool, optional
        Whether to return string if conversion fails.

    Returns:
    -------
    StrEnum
        Converted enum value.

    Raises:
    ------
    ValueError
        If conversion fails and str_default is False.
    """
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value.lower())
    except ValueError:
        if str_default:
            return cast('StrEnum', value)
        raise


def _setup_pushforward(
    *,
    pushforward_type: Pushforward | str,
    predictive_type: Predictive | str,
    pushforward_fns: list[Callable] | None,
) -> tuple[Callable, list[Callable]]:
    """Set up pushforward functions based on type and predictive method.

    Parameters
    ----------
    pushforward_type : Pushforward | str
        Type of pushforward approximation to use.
    predictive_type : Predictive | str
        Type of predictive distribution to use.
    pushforward_fns : list[Callable] | None
        Custom pushforward functions to use.

    Returns:
    -------
    tuple[Callable, list[Callable]]
        (set_pushforward, list_of_fns) according to specification.

    Raises:
    ------
    ValueError
        If pushforward or predictive type is invalid.
    """
    pushforward_type = _convert_to_enum(Pushforward, pushforward_type)
    predictive_type = _convert_to_enum(Predictive, predictive_type)
    pushforward_fns = [] if pushforward_fns is None else pushforward_fns

    if pushforward_type is Pushforward.LINEAR:
        set_pushforward = set_lin_pushforward
        if not pushforward_fns:
            pushforward_fns = [lin_setup, lin_pred_mean, lin_pred_std]
            if predictive_type is Predictive.MC_BRIDGE:
                pushforward_fns.append(lin_mc_pred_act)
            elif predictive_type in _SPECIAL_PREDICTIVES:
                pushforward_fns.append(
                    partial(
                        lin_special_pred_act,
                        special_pred_type=predictive_type,
                    )
                )
            elif predictive_type is not Predictive.NONE:
                msg = f"Invalid predictive type: {predictive_type}"
                raise ValueError(msg)

    elif pushforward_type is Pushforward.NONLINEAR:
        set_pushforward = set_nonlin_pushforward
        if not pushforward_fns:
            pushforward_fns = [nonlin_setup, nonlin_pred_mean, nonlin_pred_std]
            if predictive_type is Predictive.MC_BRIDGE:
                pushforward_fns.append(nonlin_mc_pred_act)
            elif predictive_type in _SPECIAL_PREDICTIVES:
                msg = (
                    f"{predictive_type.value} not supported for non-linear pushforward."
                )
                raise ValueError(msg)
    else:
        msg = f"Invalid pushforward type: {pushforward_type}"
        raise ValueError(msg)

    return set_pushforward, pushforward_fns


def register_calibration_method(method_name: str, method_fn: Callable) -> None:
    """Register a new calibration method.

    Parameters
    ----------
    method_name : str
        Name of the calibration method.
    method_fn : Callable
        Function implementing the calibration method.

    Notes:
    -----
    The method function should have signature:
    method_fn(objective: Callable, **kwargs) -> float
    """
    calibration_options[method_name] = method_fn
    logger.info(f"Registered new calibration method: {method_name}")


def _make_nll_objective(
    set_prob_predictive: Callable,
) -> Callable[[PriorArguments, Data], Float]:
    """Create negative log-likelihood objective for calibration.

    Parameters
    ----------
    set_prob_predictive : Callable
        Function to set up predictive distribution.

    Returns:
    -------
    Callable[[PriorArguments, Data], Float]
        JIT-compiled objective function.
    """
    return jax.jit(
        lambda prior_args, batch: evaluate_for_given_prior_arguments(
            prior_arguments=prior_args,
            data=batch,
            set_prob_predictive=set_prob_predictive,
            metric=nll_gaussian,
        )
    )


def _make_chi2_objective(
    set_prob_predictive: Callable,
) -> Callable[[PriorArguments, Data], Float]:
    """Create chi-squared objective for calibration.

    Parameters
    ----------
    set_prob_predictive : Callable
        Function to set up predictive distribution.

    Returns:
    -------
    Callable[[PriorArguments, Data], Float]
        JIT-compiled objective function.
    """
    return jax.jit(
        lambda prior_args, batch: evaluate_for_given_prior_arguments(
            prior_arguments=prior_args,
            data=batch,
            set_prob_predictive=set_prob_predictive,
            metric=chi_squared_zero,
        )
    )


def _make_ece_objective(
    set_prob_predictive: Callable,
) -> Callable[[PriorArguments, Data], Float]:
    """Create expected calibration error objective.

    Parameters
    ----------
    set_prob_predictive : Callable
        Function to set up predictive distribution.

    Returns:
    -------
    Callable[[PriorArguments, Data], Float]
        Objective function computing ECE.
    """
    def ece(*, mc_pred_act: Array, target: Array, **kwargs) -> Float:
        del kwargs
        conf = jnp.max(mc_pred_act, axis=-1)
        correct = correctness(mc_pred_act, target) * 1
        val = expected_calibration_error(
            confidence=conf,
            correctness=correct,
            num_bins=15,
        )
        return val

    return lambda prior_args, batch: evaluate_for_given_prior_arguments(
            prior_arguments=prior_args,
            data=batch,
            set_prob_predictive=set_prob_predictive,
            metric=ece,
        )


def _make_mll_objective(
    curv_estimate: PyTree,
    model_fn: ModelFn,
    params: Params,
    curv_type: CurvApprox,
    loss_fn: LossFn,
) -> Callable[[PriorArguments, Data], Float]:
    """Create marginal log-likelihood objective for calibration.

    Parameters
    ----------
    curv_estimate : PyTree
        Estimated curvature.
    model_fn : ModelFn
        Neural network forward pass.
    params : Params
        Network parameters.
    curv_type : CurvApprox
        Type of curvature approximation.
    loss_fn : LossFn
        Loss function used.

    Returns:
    -------
    Callable[[PriorArguments, Data], Float]
        JIT-compiled objective function.
    """
    return jax.jit(
        lambda prior_args, batch: -marginal_log_likelihood(
            curv_estimate=curv_estimate,
            prior_arguments=prior_args,
            data=batch,
            model_fn=model_fn,
            params=params,
            loss_fn=loss_fn,
            curv_type=curv_type,
        )
    )


def _build_calibration_objective(
    objective_type: CalibrationObjective | str,
    *,
    set_prob_predictive: Callable,
    curv_estimate: PyTree | None = None,
    model_fn: ModelFn | None = None,
    params: Params | None = None,
    curv_type: CurvApprox | None = None,
    loss_fn: LossFn,
) -> Callable[[PriorArguments, Data], Float]:
    """Build calibration objective function based on type.

    Parameters
    ----------
    objective_type : CalibrationObjective | str
        Type of calibration objective.
    set_prob_predictive : Callable
        Function to set up predictive distribution.
    curv_estimate : PyTree | None, optional
        Estimated curvature.
    model_fn : ModelFn | None, optional
        Neural network forward pass.
    params : Params | None, optional
        Network parameters.
    curv_type : CurvApprox | None, optional
        Type of curvature approximation.
    loss_fn : LossFn
        Loss function used.

    Returns:
    -------
    Callable[[PriorArguments, Data], Float]
        Calibration objective function.

    Raises:
    ------
    ValueError
        If required arguments are missing or objective type is invalid.
    """
    objective_type = _convert_to_enum(CalibrationObjective, objective_type)

    if (
        objective_type is CalibrationObjective.MARGINAL_LOG_LIKELIHOOD
        and _check_if_none(curv_estimate, model_fn, params, curv_type)
    ):
        msg = (
            "Marginal log-likelihood objective requires "
            "`curv_estimate`, `model_fn`, `params`, `curv_type`."
        )
        raise ValueError(msg)

    if model_fn is None or curv_type is None:
        msg = "model_fn and curv_type must not be None for MLL objective"
        raise ValueError(msg)

    match objective_type:
        case CalibrationObjective.NLL:
            return _make_nll_objective(set_prob_predictive)
        case CalibrationObjective.CHI_SQUARED:
            return _make_chi2_objective(set_prob_predictive)
        case CalibrationObjective.MARGINAL_LOG_LIKELIHOOD:
            return _make_mll_objective(
                curv_estimate=curv_estimate,
                model_fn=model_fn,
                params=params,
                curv_type=curv_type,
                loss_fn=loss_fn,
            )
        case CalibrationObjective.ECE:
            return _make_ece_objective(set_prob_predictive)
        case _:
            msg = f"Unknown calibration objective: {objective_type}"
            raise ValueError(msg)


def _resolve_metrics(
    metrics: DefaultMetrics | list[Callable] | str,
) -> list[Callable]:
    """Resolve metrics to list of callable functions.

    Parameters
    ----------
    metrics : DefaultMetrics | list[Callable] | str
        Metrics specification to resolve.

    Returns:
    -------
    list[Callable]
        List of metric functions.

    Raises:
    ------
    ValueError
        If metrics specification is invalid.
    """
    if isinstance(metrics, str):
        metrics = _convert_to_enum(DefaultMetrics, metrics)

    if metrics == DefaultMetrics.REGRESSION:
        return DEFAULT_REGRESSION_METRICS
    if metrics == DefaultMetrics.CLASSIFICATION:
        return [
            apply_fns(
                lambda map, **kwargs:  # noqa: ARG005
                    jnp.max(jax.nn.softmax(map, axis=-1), axis=-1),
                lambda mc_pred_act, **kwargs:  # noqa: ARG005
                    jnp.max(mc_pred_act, axis=-1),
                lambda map, target, **kwargs:  # noqa: ARG005
                    correctness(map, target) * 1,
                lambda map, target, **kwargs:  # noqa: ARG005
                    correctness(map, target) * 1,
                names=[
                    "confidences_map",
                    "confidences_pred",
                    "correctness_map",
                    "correctness_pred",
                ],
            ),
        ]
    if isinstance(metrics, (list, tuple)):
        if not metrics:
            msg = "Metrics list must not be empty."
            raise ValueError(msg)
        return list(metrics)

    msg = (
        f"Parameter `metrics` must be DefaultMetrics.REGRESSION, "
        f"DefaultMetrics.CLASSIFICATION, or a *non-empty* list of callables, "
        f"got {type(metrics).__name__}"
    )
    raise ValueError(msg)


# ------------------------------------------------------------------------------
# Main functions
# ------------------------------------------------------------------------------

def GGN(
    model_fn: ModelFn,
    params: Params,
    data: Data | Iterable,
    loss_fn: LossFn,
    *,
    factor: float = 1.0,
    has_batch: bool = True,
    verbose_logging: bool = True,
) -> Callable[[Params], Params]:
    """Create a GGN matrix-vector product function.

    Parameters
    ----------
    model_fn : ModelFn
        Neural network forward pass.
    params : Params
        Network parameters.
    data : Data | Iterable
        Training data.
    loss_fn : LossFn
        Loss function to use.
    factor : float, optional
        Scaling factor for GGN.
    has_batch : bool, optional
        Whether model expects batch dimension.
    verbose_logging : bool, optional
        Whether to enable verbose logging.

    Returns:
    -------
    Callable[[Params], Params]
        GGN matrix-vector product function.

    Raises:
    ------
    ValueError
        If input/output shapes don't match.
    """
    ggn_mv = create_ggn_mv_without_data(  # type: ignore[call-arg]
        model_fn=model_fn,
        params=params,
        loss_fn=loss_fn,
        factor=factor,
        has_batch=has_batch,
    )

    mv_bound = _maybe_wrap_loader_or_batch(
        ggn_mv,
        data,
        loader_kwargs={
            "verbose_logging": verbose_logging,
        }
    )

    test = mv_bound(params)
    if not jax.tree.all(
        jax.tree.map(lambda x, y: x.shape == y.shape, test, params),
    ):
        msg = "Setup of GGN-MV failed: input and output shapes do not match."
        raise ValueError(msg)

    return mv_bound


def laplace(
    model_fn: ModelFn,
    params: Params,
    data: Data | Iterable,
    *,
    loss_fn: LossFn,
    curv_type: CurvApprox,
    num_curv_samples: Int = 1,
    num_total_samples: Int = 1,
    has_batch: bool = True,
    **curv_kwargs,
) -> tuple[Callable[[PriorArguments, Float], Posterior], PyTree]:
    """Estimate curvature & obtain a Gaussian weight-space posterior.

    This function computes a Laplace approximation to the posterior distribution over
    neural network weights. It estimates the curvature of the loss landscape and
    constructs a Gaussian approximation centered at the MAP estimate.

    Parameters
    ----------
    model_fn : ModelFn
        The neural network forward pass function that takes input and parameters.
    params : Params
        The MAP estimate of the network parameters.
    data : Data | Iterable
        Either a single batch (tuple/dict) or a DataLoader-like iterable containing
        the training data.
    loss_fn : LossFn
        The supervised loss function to use (e.g., "mse" for regression).
    curv_type : CurvApprox
        Type of curvature approximation to use (e.g., "ggn", "diag-ggn").
    num_curv_samples : Int, optional
        Number of Monte Carlo samples used to estimate the GGN, by default 1.
    num_total_samples : Int, optional
        Total number of samples in the dataset, by default 1.
    has_batch : bool, optional
        Whether the model expects a leading batch axis, by default True.
    **curv_kwargs
        Additional arguments forwarded to the curvature estimation function.

    Returns:
    -------
    tuple[Callable[[PriorArguments, Float], Posterior], PyTree]
        A tuple containing:
        - posterior_fn : Callable
            Function that generates samples from the posterior given prior arguments.
        - curv_estimate : PyTree
            The estimated curvature in the chosen representation.

    Notes:
    -----
    The function supports different curvature approximations:
    - Full GGN: Computes the full Generalized Gauss-Newton matrix
    - Diagonal GGN: Approximates the GGN with its diagonal
    - Low-rank GGN: Uses Lanczos or LOBPCG for efficient approximation
    """
    # Convert curv_type to enum
    curv_type_enum = _convert_to_enum(CurvApprox, curv_type)

    # Calculate factor
    factor = float(num_curv_samples) / float(num_total_samples)
    logger.debug(
        "Creating curvature MV - factor = {}/{} = {}",
        num_curv_samples,
        num_total_samples,
        factor,
    )

    # Set GGN MV
    ggn_mv = GGN(
        model_fn,
        params,
        data,
        loss_fn=loss_fn,
        factor=factor,
        has_batch=has_batch,
    )

    # Curvature estimation
    curv_estimate = estimate_curvature(
        curv_type=curv_type_enum,
        mv=ggn_mv,
        layout=params,
        **curv_kwargs,
    )
    logger.debug("Curvature estimated: {}", curv_type_enum)

    # Posterior (Gaussian)
    posterior_fn = set_posterior_fn(
        curv_type=curv_type_enum,
        curv_estimate=curv_estimate,
        layout=params,
        **curv_kwargs,
    )
    logger.debug("Posterior callable constructed.")

    return posterior_fn, curv_estimate


def calibration(
    posterior_fn: Callable[[PriorArguments, Float], Posterior],
    model_fn: ModelFn,
    params: Params,
    data: Data,
    *,
    loss_fn: LossFn,
    curv_estimate: PyTree,
    curv_type: CurvApprox,
    predictive_type: Predictive | str = Predictive.NONE,
    pushforward_type: Pushforward | str = Pushforward.LINEAR,
    pushforward_fns: list[Callable] | None = None,
    sample_key: KeyType = DEFAULT_KEY,
    num_samples: int = 30,
    calibration_objective: CalibrationObjective | str = CalibrationObjective.NLL,
    calibration_method: CalibrationMethod | str = CalibrationMethod.GRID_SEARCH,
    **calibration_kwargs,
) -> tuple[PriorArguments, Callable[[InputArray], dict[str, Array]]]:
    """Calibrate hyperparameters of the Laplace approximation.

    This function tunes the prior precision (or similar hyperparameters) of the Laplace
    approximation by optimizing a specified objective function. It supports different
    calibration objectives and methods.

    Parameters
    ----------
    posterior_fn : Callable[[PriorArguments, Float], Posterior]
        Function that generates samples from the posterior.
    model_fn : ModelFn
        The neural network forward pass function.
    params : Params
        The MAP estimate of the network parameters.
    data : Data
        The validation data used for calibration.
    loss_fn : LossFn
        The supervised loss function used for training.
    curv_estimate : PyTree
        The estimated curvature from the Laplace approximation.
    curv_type : CurvApprox
        Type of curvature approximation used.
    predictive_type : Predictive | str, optional
        Type of predictive distribution to use, by default Predictive.NONE.
    pushforward_type : Pushforward | str, optional
        Type of pushforward approximation to use, by default Pushforward.LINEAR.
    pushforward_fns : list[Callable] | None, optional
        Custom pushforward functions to use, by default None.
    calibration_objective : CalibrationObjective | str, optional
        Objective function to optimize during calibration, by default CalibrationObjective.NLL.
    calibration_method : CalibrationMethod | str, optional
        Method to use for calibration, by default CalibrationMethod.GRID_SEARCH.
    **calibration_kwargs
        Additional arguments for the calibration method.

    Returns:
    -------
    tuple[PriorArguments, Callable[[InputArray], dict[str, Array]]]
        A tuple containing:
        - prior_arguments : PriorArguments
            Dictionary of calibrated hyperparameters.
        - set_prob_predictive : Callable
            Function that creates a predictive distribution given prior arguments.

    Notes:
    -----
    Supported calibration objectives:
    - NLL: Negative log-likelihood
    - CHI_SQUARED: Chi-squared statistic
    - MARGINAL_LOG_LIKELIHOOD: Marginal log-likelihood
    - ECE: Expected Calibration Error

    Supported calibration methods:
    - GRID_SEARCH: Grid search over prior precision
    """
    # Pushforward construction
    set_pushforward, pushforward_fns = _setup_pushforward(
        pushforward_type=pushforward_type,
        predictive_type=predictive_type,
        pushforward_fns=pushforward_fns,
    )

    set_prob_predictive = partial(
        set_pushforward,
        model_fn=model_fn,
        mean_params=params,
        posterior_fn=posterior_fn,
        pushforward_fns=pushforward_fns,
        key=sample_key,
        num_samples=num_samples,
    )

    # Calibration objective & optimisation
    objective_fn = _build_calibration_objective(
        objective_type=calibration_objective,
        set_prob_predictive=set_prob_predictive,
        curv_estimate=curv_estimate,
        model_fn=model_fn,
        params=params,
        loss_fn=loss_fn,
        curv_type=curv_type,
    )

    calibration_method = _convert_to_enum(
        CalibrationMethod, calibration_method, str_default=True
    )

    if calibration_method == CalibrationMethod.GRID_SEARCH:
        # Get default values if not provided
        log_prior_prec_min = calibration_kwargs.get("log_prior_prec_min", -3.0)
        log_prior_prec_max = calibration_kwargs.get("log_prior_prec_max", 3.0)
        grid_size = calibration_kwargs.get("grid_size", 50)
        patience = calibration_kwargs.get("patience", 5)

        # Transform calibration batch to {"input": ..., "target": ...}
        data = _validate_and_get_transform(data)(data)

        logger.debug(
            "Starting calibration with objective {} on grid [{}, {}] ({} pts, pat={})",
            calibration_objective,
            log_prior_prec_min,
            log_prior_prec_max,
            grid_size,
            patience,
        )

        def objective(x):
            return objective_fn(x, data)

        prior_prec = calibration_options[calibration_method](
            objective=objective,
            log_prior_prec_min=log_prior_prec_min,
            log_prior_prec_max=log_prior_prec_max,
            grid_size=grid_size,
            patience=patience,
        )
        prior_args = {"prior_prec": prior_prec}

    elif calibration_method in calibration_options:
        data = _validate_and_get_transform(data)(data)
        prior_args = calibration_options[calibration_method](
            objective=objective_fn,
            data=data,
            **calibration_kwargs,
        )
    else:
        msg = f"Unknown calibration method: {calibration_method}"
        raise ValueError(msg)
    logger.debug("Calibrated prior args = {}", prior_args)

    return prior_args, set_prob_predictive


def evaluation(
    posterior_fn: Callable[[PriorArguments, Float], Posterior],
    model_fn: ModelFn,
    params: Params,
    arguments: PriorArguments,
    data: Data,
    *,
    metrics: DefaultMetrics | list[Callable] | str = DefaultMetrics.REGRESSION,
    predictive_type: Predictive | str = Predictive.NONE,
    pushforward_type: Pushforward | str = Pushforward.LINEAR,
    pushforward_fns: list[Callable] | None = None,
    reduce: Callable = identity,
    sample_key: KeyType = DEFAULT_KEY,
    num_samples: int = 10,
) -> tuple[dict[str, Array], Callable[[InputArray], dict[str, Array]]]:
    """Evaluate the calibrated Laplace approximation.

    This function assesses the performance of the calibrated Laplace approximation
    by computing various metrics on the test data. It supports both regression and
    classification tasks with different predictive distributions.

    Parameters
    ----------
    posterior_fn : Callable[[PriorArguments, Float], Posterior]
        Function that generates samples from the posterior.
    model_fn : ModelFn
        The neural network forward pass function.
    params : Params
        The MAP estimate of the network parameters.
    arguments : PriorArguments
        The calibrated prior arguments.
    data : Data
        The test data for evaluation.
    metrics : DefaultMetrics | list[Callable] | str, optional
        Metrics to compute during evaluation, by default DefaultMetrics.REGRESSION.
    predictive_type : Predictive | str, optional
        Type of predictive distribution to use, by default Predictive.NONE.
    pushforward_type : Pushforward | str, optional
        Type of pushforward approximation to use, by default Pushforward.LINEAR.
    pushforward_fns : list[Callable] | None, optional
        Custom pushforward functions to use, by default None.
    reduce : Callable, optional
        Function to reduce metrics across batches, by default identity.
    sample_key : KeyType, optional
        Random key for sampling, by default jax.random.key(0).
    num_samples : int, optional
        Number of samples for Monte Carlo predictions, by default 10.

    Returns:
    -------
    tuple[dict[str, Array], Callable[[InputArray], dict[str, Array]]]
        A tuple containing:
        - results : dict
            Dictionary of computed metrics.
        - prob_predictive : Callable
            The predictive distribution function.

    Notes:
    -----
    Supported metrics:
    - REGRESSION: Default metrics for regression tasks
    - CLASSIFICATION: Default metrics for classification tasks
    - Custom metrics can be provided as a list of callables

    The function supports both linearized and Monte Carlo predictions through
    different pushforward types.
    """
    metrics_list = _resolve_metrics(metrics)

    set_pushforward, pushforward_fns = _setup_pushforward(
        pushforward_type=pushforward_type,
        predictive_type=predictive_type,
        pushforward_fns=pushforward_fns,
    )

    # Build predictive distribution
    prob_predictive = set_pushforward(
        prior_arguments=arguments,
        model_fn=model_fn,
        mean_params=params,
        posterior_fn=posterior_fn,
        pushforward_fns=pushforward_fns,
        key=sample_key,
        num_samples=num_samples,
    )

    # Evaluate
    batch = _validate_and_get_transform(data)(data)
    results = evaluate_metrics_on_dataset(
        pred_fn=prob_predictive,
        data=batch,
        metrics=metrics_list,
        reduce=reduce,
    )

    return results, prob_predictive
