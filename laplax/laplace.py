"""Functional API around Laplax's Laplace approximation stack.

Public entry points
-------------------
* :func:`laplace`     - fit curvature / posterior fn
* :func:`calibration` - tune prior precision (or similar args)
* :func:`evaluation`  - evaluate performance of calibrated model
"""

from enum import StrEnum
from functools import partial

import jax
import jax.numpy as jnp
from loguru import logger

# Laplax imports
from laplax.curv.cov import estimate_curvature, set_posterior_fn
from laplax.curv.ggn import create_ggn_mv_without_data
from laplax.enums import LossFn
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
from laplax.eval.utils import evaluate_metrics_on_dataset
from laplax.types import (
    Any,
    Callable,
    CurvApprox,
    Data,
    Int,
    Iterable,
    KeyType,
    ModelFn,
    Params,
    PriorArguments,
)
from laplax.util.loader import (
    DataLoaderMV,
    identity,
    input_target_split,
    reduce_add,
)

# ------------------------------------------------------------------------------
# API specific enumerations
# ------------------------------------------------------------------------------


def _convert_to_enum(enum_cls, value, *, str_default=False):
    """Convert string to enum, pass through if already enum."""
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value.lower())
    except ValueError:
        if str_default:
            return value
        raise


class CalibrationObjective(StrEnum):
    """Supported calibration objectives (minimisation!)."""

    NLL = "nll"
    CHI_SQUARED = "chi_squared"
    MARGINAL_LOG_LIKELIHOOD = "marginal_log_likelihood"
    ECE = "ece"


class CalibrationMethod(StrEnum):
    GRID_SEARCH = "grid_search"


calibration_options = {
    CalibrationMethod.GRID_SEARCH: optimize_prior_prec,
}


def register_calibration_method(method_name: str, method_fn: Callable):
    """Register a new calibration method.

    Parameters
    ----------
    method_name : str
        Name of the calibration method. This will be added to the CalibrationMethod enum.
    method_fn : Callable
        Function implementing the calibration method. Should have the signature:
        method_fn(objective: Callable, **kwargs) -> float
        where objective is a function that takes a prior_prec value and returns a scalar loss.

    Returns
    -------
    None
        The method is registered in the calibration_options dictionary.
 
    Examples
    --------
    >>> def my_custom_calibration(objective, **kwargs):
    ...     # Custom implementation
    ...     return optimal_prior_prec
    >>> register_calibration_method("my_method", my_custom_calibration)
    >>> # Now you can use it in calibration:
    >>> calibration(..., calibration_method="my_method")
    """
    # Register the method in the options dictionary
    calibration_options[method_name] = method_fn

    logger.info(f"Registered new calibration method: {method_name}")


class PushforwardType(StrEnum):
    LINEAR = "linear"
    NONLINEAR = "nonlinear"


class PredictiveType(StrEnum):
    MC_BRIDGE = "mc_bridge"
    LAPLACE_BRIDGE = "laplace_bridge"
    MEAN_FIELD_0 = "mean_field_0"
    MEAN_FIELD_1 = "mean_field_1"
    MEAN_FIELD_2 = "mean_field_2"
    NONE = "none"


_SPECIAL_PREDICTIVE_TYPES = {
    PredictiveType.LAPLACE_BRIDGE,
    PredictiveType.MEAN_FIELD_0,
    PredictiveType.MEAN_FIELD_1,
    PredictiveType.MEAN_FIELD_2,
}


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def _validate_and_get_transform(batch: Data | Any) -> Callable[[Any], Data]:
    """Return the transform that converts a *single* batch to (inputs, targets).

    Raises:
    ------
    ValueError
        If `batch` is not a tuple / dict or misses the `'input'/'target'` keys.
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

# ------------------------------------------------------------------------------
# GGN API
# ------------------------------------------------------------------------------


EMPTY_DICT = {}


def _maybe_wrap_loader_or_batch(
    mv_fn: Callable[..., Any],
    data: Data | Iterable,
    *,
    loader_kwargs: dict = EMPTY_DICT,
) -> Callable[..., Any]:
    """If `data` is an iterable (loader) wrap `mv_fn`; else fix the inputs."""
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


def GGN(
    model_fn: ModelFn,
    params: Params,
    data: Data | Iterable,
    loss_fn: LossFn,
    *,
    factor: float = 1.0,
    has_batch: bool = True,
    verbose_logging: bool = True,
):
    """Create a GGN MV from a model and data.

    Parameters
    ----------
    model_fn : ModelFn
        The model function.
    params : Params
        The parameters of the model.
    data : Data | Iterable
        The data to be used for the GGN MV.
    loss_fn : LossFn
        The loss function to be used for the GGN MV.
    factor : float
        The factor to be used for the GGN MV.
    has_batch : bool
        Whether the model expects a leading batch axis.

    Returns:
    -------
    mv_bound : Callable
        The GGN MV.
    """
    ggn_mv = create_ggn_mv_without_data(  # type: ignore[call-arg]
        model_fn=model_fn,
        params=params,
        loss_fn=loss_fn,
        factor=factor,
        has_batch=has_batch,
    )

    # Bind data / streaming loader
    mv_bound = _maybe_wrap_loader_or_batch(
        ggn_mv,
        data,
        loader_kwargs={
            "verbose_logging": verbose_logging,
        }
    )

    # Test input and output shapes
    test = mv_bound(params)
    if not jax.tree.all(
        jax.tree.map(lambda x, y: x.shape == y.shape, test, params),
    ):
        msg = "Setup of GGN-MV failed: input and output shapes do not match."
        raise ValueError(msg)

    return mv_bound

# ------------------------------------------------------------------------------
# laplace (fit curvature)
# ------------------------------------------------------------------------------


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
):
    """Estimate curvature & obtain a Gaussian weight-space posterior.

    Parameters
    ----------
    model_fn, params
        Network forward pass and its parameters.
    data
        Either a *single* batch (tuple/dict) **or** a `DataLoader`-like iterable.
    loss_fn
        Supervised loss function (e.g. ``"mse"``).
    curv_type
        Curvature approximation to use (`ggn`, `diag-ggn`, …).
    num_curv_samples, num_total_samples
        Number of MC samples used to estimate GGN / total samples in dataset.
    has_batch
        Whether the model expects a leading batch axis.
    **curv_kwargs
        Forwarded to :func:`laplax.curv.cov.estimate_curvature`.

    Returns:
    -------
    posterior_fn
        Callable `(params_mean, rng_key, **posterior_kwargs) -> sample`.
    curv_estimate
        The curvature estimate in the chosen representation.
    """
    # Convert curv_type to enum
    curv_type = _convert_to_enum(CurvApprox, curv_type)
    loss_fn = _convert_to_enum(LossFn, loss_fn)

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
        curv_type=curv_type,
        mv=ggn_mv,
        layout=params,
        **curv_kwargs,
    )
    logger.debug("Curvature estimated: {}", curv_type)

    # Posterior (Gaussian)
    posterior_fn = set_posterior_fn(
        curv_type=curv_type,
        curv_estimate=curv_estimate,
        layout=params,
        **curv_kwargs,
    )
    logger.debug("Posterior callable constructed.")

    return posterior_fn, curv_estimate


# ------------------------------------------------------------------------------
# calibration helpers
# ------------------------------------------------------------------------------


def _make_nll_objective(set_prob_predictive: Callable) -> Callable:
    return jax.jit(
        lambda prior_args, batch: evaluate_for_given_prior_arguments(
            prior_arguments=prior_args,
            data=batch,
            set_prob_predictive=set_prob_predictive,
            metric=nll_gaussian,
        )
    )


def _make_chi2_objective(set_prob_predictive: Callable) -> Callable:
    return jax.jit(
        lambda prior_args, batch: evaluate_for_given_prior_arguments(
            prior_arguments=prior_args,
            data=batch,
            set_prob_predictive=set_prob_predictive,
            metric=chi_squared_zero,
        )
    )


def _make_ece_objective(set_prob_predictive: Callable) -> Callable:
    def ece(**kwargs):
        mc_pred_act = kwargs["mc_pred_act"]
        target = kwargs["target"]
        conf = jnp.max(mc_pred_act, axis=-1)
        correct = correctness(mc_pred_act, target) * 1
        val = expected_calibration_error(
            confidence=conf,
            correctness=correct,
            num_bins=15,
        )
        return val

    # return jax.jit(
    return lambda prior_args, batch: evaluate_for_given_prior_arguments(
            prior_arguments=prior_args,
            data=batch,
            set_prob_predictive=set_prob_predictive,
            metric=ece,
        )
    # )


def _make_mll_objective(
    curv_estimate,
    model_fn: ModelFn,
    params: Params,
    curv_type: CurvApprox,
    loss_fn: LossFn,
):
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


def _check_if_none(*args: Any) -> bool:
    return any(x is None for x in args)


def _build_calibration_objective(
    objective_type: CalibrationObjective | str,
    *,
    set_prob_predictive: Callable,
    curv_estimate=None,
    model_fn=None,
    params=None,
    curv_type=None,
    loss_fn: LossFn,
) -> Callable:
    """Factory selecting / validating the requested calibration objective."""
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


# ------------------------------------------------------------------------------
# pushforward helpers
# ------------------------------------------------------------------------------


def _setup_pushforward(
    *,
    pushforward_type: PushforwardType | str,
    predictive_type: PredictiveType | str,
    pushforward_fns: list[Callable] | None,
):
    """Return `(set_pushforward, list_of_fns)` according to user specification."""
    pushforward_type = _convert_to_enum(PushforwardType, pushforward_type)
    predictive_type = _convert_to_enum(PredictiveType, predictive_type)
    pushforward_fns = [] if pushforward_fns is None else pushforward_fns

    if pushforward_type is PushforwardType.LINEAR:
        set_pushforward = set_lin_pushforward
        if not pushforward_fns:
            pushforward_fns = [lin_setup, lin_pred_mean, lin_pred_std]
            if predictive_type is PredictiveType.MC_BRIDGE:
                pushforward_fns.append(lin_mc_pred_act)
            elif predictive_type in _SPECIAL_PREDICTIVE_TYPES:
                pushforward_fns.append(
                    partial(
                        lin_special_pred_act,
                        special_pred_type=predictive_type,
                    )
                )
            elif predictive_type is not PredictiveType.NONE:
                msg = f"Invalid predictive type: {predictive_type}"
                raise ValueError(msg)

    elif pushforward_type is PushforwardType.NONLINEAR:
        set_pushforward = set_nonlin_pushforward
        if not pushforward_fns:
            pushforward_fns = [nonlin_setup, nonlin_pred_mean, nonlin_pred_std]
            if predictive_type is PredictiveType.MC_BRIDGE:
                pushforward_fns.append(nonlin_mc_pred_act)
            elif predictive_type in _SPECIAL_PREDICTIVE_TYPES:
                msg = (
                    f"{predictive_type.value} not supported for non-linear pushforward."
                )
                raise ValueError(msg)
    else:
        msg = f"Invalid pushforward type: {pushforward_type}"
        raise ValueError(msg)

    return set_pushforward, pushforward_fns


# ------------------------------------------------------------------------------
# calibration
# ------------------------------------------------------------------------------


def calibration(
    posterior_fn: Callable,
    model_fn: ModelFn,
    params: Params,
    data: Data,
    *,
    loss_fn: LossFn,
    curv_estimate,
    curv_type,
    predictive_type: PredictiveType | str = PredictiveType.NONE,
    pushforward_type: PushforwardType | str = PushforwardType.LINEAR,
    pushforward_fns: list[Callable] | None = None,
    calibration_objective: CalibrationObjective | str = CalibrationObjective.NLL,
    calibration_method: CalibrationMethod | str = CalibrationMethod.GRID_SEARCH,
    **calibration_kwargs,
):
    """Calibrate a *single* scalar prior precision (or similar parameter).

    Returns:
    -------
    prior_arguments
        Dict that can be forwarded to predictive functions, e.g. ``{"prior_prec": …}``.
    set_prob_predictive
        Callable that turns `prior_arguments` into a predictive distribution.
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
        key=jax.random.key(0),  # deterministic seed (override outside if needed)
        num_samples=30,
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
        prior_prec = calibration_options[calibration_method](
            objective=partial(objective_fn, batch=data),
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


# ------------------------------------------------------------------------------
# evaluation helpers
# ------------------------------------------------------------------------------


def _resolve_metrics(metrics: str | list[Callable]) -> list[Callable]:
    if metrics == "regression":
        return DEFAULT_REGRESSION_METRICS
    if metrics == "classification":
        msg = "Classification metrics are not yet implemented."
        raise NotImplementedError(msg)
    if isinstance(metrics, (list, tuple)):
        if not metrics:
            msg = "Metrics list must not be empty."
            raise ValueError(msg)
        return list(metrics)

    msg = (
        "Parameter `metrics` must be 'regression', 'classification', or "
        "a *non-empty* list of callables."
    )
    raise ValueError(msg)


# ------------------------------------------------------------------------------
# evaluation
# ------------------------------------------------------------------------------


def evaluation(
    posterior_fn: Callable,
    model_fn: ModelFn,
    params: Params,
    arguments: PriorArguments,
    data: Data,
    *,
    metrics: str | list[Callable] = "regression",
    predictive_type: PredictiveType | str = PredictiveType.NONE,
    pushforward_type: PushforwardType | str = PushforwardType.LINEAR,
    pushforward_fns: list[Callable] | None = None,
    reduce: Callable = identity,
    sample_key: KeyType = 0,
    num_samples: int = 10,
):
    """Run predictive evaluation after calibration (or fixed prior args)."""
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

    # Log results.
    logger.debug(
        "Evaluation finished with metrics: {}",
        results if reduce == identity else jax.tree.map(jnp.mean, results)
    )
    return results, prob_predictive
