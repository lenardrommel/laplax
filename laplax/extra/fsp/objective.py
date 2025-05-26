import jax
from jax import numpy as jnp
from laplax.enums import LossFn
from laplax.types import Callable, Data, Float, Int, ModelFn, Params, PredArray


def create_loss_nll(
    model_fn: ModelFn,
    dataset_size: int | None = None,
):
    r"""Create the NLL loss function for FSP training.

    Computes the negative log-likelihood:
    $$
    -\log p(y | f(X)) = -\sum_i \log \mathcal{N}(y_i | f(x_i), \sigma^2)
    $$
    """

    def loss_nll(
        data: Data, params: Params, scale: Float | Params | None = None
    ) -> Float:
        preds = jax.vmap(model_fn, in_axes=(0, None))(data["input"], params)
        nll = -jax.scipy.stats.norm.logpdf(
            data["target"], loc=preds, scale=scale
        ).mean()
        return dataset_size * nll

    return loss_nll


def create_loss_reg(
    model_fn: ModelFn,
    prior_mean: PredArray,
    prior_cov_kernel: Callable[[PredArray, PredArray], Float],
):
    r"""Create the FSP regularization loss function.

    Computes the RKHS regularization:
    $$
    \frac{1}{2} (f(c) - m)^T K^{-1}(c, c) (f(c) - m)
    $$
    """

    def loss_reg(context_points: PredArray, params: Params) -> Float:
        f_c = jax.vmap(model_fn, in_axes=(0, None))(context_points, params) - prior_mean
        K_c_c = prior_cov_kernel(context_points)
        left = jax.numpy.linalg.solve(K_c_c, f_c)
        return 0.5 * jax.numpy.einsum("ij,ij->", f_c, left)

    return loss_reg


def create_fsp_objective(
    model_fn: ModelFn,
    dataset_size: Int,
    prior_mean: PredArray,
    prior_cov_kernel: Callable,
):
    """Create FSP objective combining NLL and regularization losses."""
    loss_nll = create_loss_nll(model_fn, dataset_size)
    loss_reg = create_loss_reg(model_fn, prior_mean, prior_cov_kernel)

    def fsp_objective(
        data: Data,
        context_points: PredArray,
        params: Params,
        scale: Float | Params | None = None,
    ) -> Float:
        nll_term = loss_nll(data, params, scale)
        reg_term = loss_reg(context_points, params)
        return nll_term + reg_term

    return fsp_objective


def select_context_points(
    n_context_points: int,
    context_selection: str,
    context_points_maxval: list[float],
    context_points_minval: list[float],
    datapoint_shape: tuple[int, ...],
    key: jax.random.PRNGKey,
):
    D = datapoint_shape[-1]

    scaled_max = jnp.array(context_points_maxval)
    scaled_min = jnp.array(context_points_minval)

    if context_selection == "random":
        window_len = (scaled_max - scaled_min) / 4.0

        start = jax.random.uniform(
            key,
            shape=(D,),
            minval=scaled_min,
            maxval=scaled_max - window_len,
        )

        w_min = start
        w_max = start + window_len

        context_points = jax.random.uniform(
            key,
            shape=(n_context_points, D),
            minval=w_min,
            maxval=w_max,
        )

    elif context_selection == "grid":
        if D == 1:
            context_points = jnp.linspace(
                context_points_minval[0], context_points_maxval[0], n_context_points
            ).reshape(-1, 1)
        else:
            raise NotImplementedError("Grid for D>1")

    else:
        raise ValueError(f"Unknown context_selection={context_selection!r}")

    return context_points
