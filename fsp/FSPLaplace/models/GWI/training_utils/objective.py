import jax

import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp

from jax.experimental import host_callback
from functools import partial


@partial(jax.jit, static_argnums=(3,4,10))
def n_gelbo_gaussian_objective(
    mean_params,
    L_params,
    ll_rho,
    model,
    prior,
    x,
    y,
    x_s,
    inducing_points,
    key,
    n_samples
):
    """
    Generalized functional ELBO objective for Gaussian likelihoods.
     
    params:
    - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
    - L_params (jax.tree_util.pytree): pre-activated variance parameters of the BNN.
    - ll_rho (float): pre-activated variance parameter of the likelihood.
    - model (Model): stochastic neural networks.
    - prior (Prior): prior distribution on context points
    - x (jnp.ndarray): input data used to calculate the expected log likelihood in the ELBO.
    - y (jnp.ndarray): targets used to calculate the expected log likelihoodin the ELBO.
    - x_sampled_context (jnp.ndarray): context points sampled from feature distribution.
    - key (jax.random.PRNGKey): random key.
    - kl_gamma (float): KL divergence regularization coefficient.
    - n_samples (int): total number of training samples.
    - n_context_points (int): total number of context points.

    returns:
    - neg_gelbo_objective (float): generalized ELBO.
    - expected_ll (float): expected log likelihood.
    - reg_kl (float): regularized KL divergence.
    """   
    # Compute W2 distance
    p_mean, p_var = prior.mean_var(x)
    p_cross_cov = prior.cross_covariance(x, x_s)
    q_mean, q_var = model.f_diag_distribution(mean_params, L_params, prior, inducing_points, x, key)
    q_cross_cov = model.f_distribution_kernel(L_params, prior, inducing_points, x, x_s)
    w2 = wasserstein2_distance(
        q_mean[:,0], p_mean[:,0], q_var[:,0], p_var[:,0], q_cross_cov[:,:,0], p_cross_cov[:,:,0]
    )

    # Compute expected log likelihood
    ll_scale = jax.nn.softplus(ll_rho)
    mean, diag_cov = model.f_diag_distribution(mean_params, L_params, prior, inducing_points, x, key)
    expected_ll = jsp.stats.norm.logpdf(y, loc=mean, scale=ll_scale).sum() 
    expected_ll -= 0.5 * diag_cov.sum() / ll_scale**2
    expected_ll *= n_samples / x.shape[0]

    # Compute generalized ELBO
    gelbo = expected_ll - w2

    return (
        -gelbo,
        {"expected_ll": expected_ll, "w2": w2, "gelbo": gelbo}
    )


@partial(jax.jit, static_argnums=(2,3,9,10))
def n_gelbo_categorical_objective(
    mean_params,
    L_params,
    model,
    prior,
    x,
    y,
    x_s,
    inducing_points,
    key,
    n_samples, 
    mc_samples
): 
    """
    Generalized functional ELBO objective for Gaussian likelihoods.
    
    params:
    - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
    - L_params (jax.tree_util.pytree): pre-activated variance parameters of the BNN.
    - model (Model): stochastic neural networks.
    - prior (Prior): prior distribution on context points
    - x (jnp.ndarray): input data used to calculate the expected log likelihood in the ELBO.
    - y (jnp.ndarray): targets used to calculate the expected log likelihoodin the ELBO.
    - x_sampled_context (jnp.ndarray): context points sampled from feature distribution.
    - key (jax.random.PRNGKey): random key.
    - mc_samples (float): number of Monte Carlo samples to estimate expected log likelihood.
    - kl_gamma (float): KL divergence regularization coefficient.
    - n_samples (int): total number of training samples.
    - n_context_points (int): total number of context points.

    returns:
    - neg_gelbo_objective (float): generalized ELBO.
    - expected_ll (float): expected log likelihood.
    - reg_kl (float): regularized KL divergence.
    """
    # Compute W2 distance
    p_mean, p_var = prior.mean_var(x)
    p_cross_cov = prior.cross_covariance(x, x_s)
    q_mean, q_var = model.f_diag_distribution(mean_params, L_params, prior, inducing_points, x, key)
    q_cross_cov = model.f_distribution_kernel(L_params, prior, inducing_points, x, x_s)
    w2 = 0
    for k in range(q_mean.shape[-1]): 
        w2 += wasserstein2_distance(
            q_mean[:,k], p_mean[:,k], q_var[:,k], p_var[:,k], q_cross_cov[:,:,k], p_cross_cov[:,:,k]
        )

    # Compute expected log likelihood
    f = model.predict_f(mean_params, L_params, prior, inducing_points, x, key, mc_samples)
    one_hot_y = jax.nn.one_hot(y.reshape(-1), num_classes=f.shape[-1])
    expected_ll = n_samples * jnp.mean(
        jnp.sum(
            one_hot_y * jax.nn.log_softmax(f, axis=-1), # (n_samples, n_batch, n_classes)
            axis=-1
        ), # (n_samples, n_batch)
        axis=0
    ).mean()

    # Compute generalized ELBO
    gelbo = expected_ll - w2

    return (
        -gelbo,
        {"expected_ll": expected_ll, "w2": w2, "gelbo": gelbo}
    )


@jax.jit
def wasserstein2_distance(
    q_mean, 
    p_mean, 
    q_var,
    p_var,
    q_cross_cov, 
    p_cross_cov
):
    """
    Compute the Wasserstein distance between two Gaussian distributions.
    
    params:
    - q_mean (jnp.ndarray): mean of the first Gaussian distribution.
    - p_mean (jnp.ndarray): mean of the second Gaussian distribution.
    - q_cov (jnp.ndarray): covariance of the first Gaussian distribution.
    - p_cov (jnp.ndarray): covariance of the second Gaussian distribution.
    
    returns:
    - wasserstein_distance (jnp.ndarray): Wasserstein distance between the two distributions.
    """
    jitter = 1e2 * jnp.eye(q_cross_cov.shape[1])
    rk_hat = q_cross_cov.T @ p_cross_cov
    eigs = eig(rk_hat + jitter)[0].real
    eigs = jnp.abs(eigs)
    eigs = eigs - jnp.diagonal(jitter)
    eigs = jnp.where(eigs > 0, eigs, 0)
    trace_q_cov_p_cov = jnp.sum(eigs**0.5) / (jnp.sqrt(np.prod(q_cross_cov.shape)))
    
    w2 = jnp.linalg.norm(q_mean - p_mean)**2 / q_mean.shape[0]
    w2 += (q_var.sum() + p_var.sum()) / q_mean.shape[0]
    w2 -= 2 * trace_q_cov_p_cov
    
    return w2.real


@jax.custom_vjp
def eig(matrix, eps=1e-6):
    """Wraps `jnp.linalg.eig` in a jit-compatible, differentiable manner.

    The custom vjp allows gradients with resepct to the eigenvectors, unlike the
    standard jax implementation of `eig`. We use an expression for the gradient
    given in [2019 Boeddeker] along with a regularization scheme used in [2021
    Colburn]. The method effectively applies a Lorentzian broadening to a term
    containing the inverse difference of eigenvalues.

    [2019 Boeddeker] https://arxiv.org/abs/1701.00392
    [2021 Coluburn] https://www.nature.com/articles/s42005-021-00568-6

    Args:
        matrix: The matrix for which eigenvalues and eigenvectors are sought.
        eps: Parameter which determines the degree of broadening.

    Returns:
        The eigenvalues and eigenvectors.
    """
    del eps
    return _eig_host(matrix)


def _eig_host(matrix: jnp.ndarray):
    """Wraps jnp.linalg.eig so that it can be jit-ed on a machine with GPUs."""
    eigenvalues_shape = jax.ShapeDtypeStruct(matrix.shape[:-1], complex)
    eigenvectors_shape = jax.ShapeDtypeStruct(matrix.shape, complex)

    def _eig_cpu(matrix: jnp.ndarray):
        # We force this computation to be performed on the cpu by jit-ing and
        # explicitly specifying the device.
        with jax.default_device(jax.devices("cpu")[0]):
            return jax.jit(jnp.linalg.eig)(matrix)

    return host_callback.call(
        _eig_cpu,
        matrix.astype(complex),
        result_shape=(eigenvalues_shape, eigenvectors_shape),
    )



def _eig_fwd(
    matrix: jnp.ndarray,
    eps: float,
):
    """Implements the forward calculation for `eig`."""
    eigenvalues, eigenvectors = _eig_host(matrix)
    return (eigenvalues, eigenvectors), (eigenvalues, eigenvectors, eps)


def _eig_bwd(
    res,
    grads,
):
    """Implements the backward calculation for `eig`."""
    eigenvalues, eigenvectors, eps = res
    grad_eigenvalues, grad_eigenvectors = grads

    # Compute the F-matrix, from equation 5 of [2021 Colburn]. This applies a
    # Lorentzian broadening to the matrix `f = 1 / (eigenvalues[i] - eigenvalues[j])`.
    eigenvalues_i = eigenvalues[..., jnp.newaxis, :]
    eigenvalues_j = eigenvalues[..., :, jnp.newaxis]
    f_broadened = jnp.divide(eigenvalues_i - eigenvalues_j, (eigenvalues_i - eigenvalues_j) ** 2 + eps)

    # Manually set the diagonal elements to zero, as we do not use broadening here.
    i = jnp.arange(f_broadened.shape[-1])
    f_broadened = f_broadened.at[..., i, i].set(0)

    # By jax convention, gradients are with respect to the complex parameters, not with
    # respect to their conjugates. Take the conjugates.
    grad_eigenvalues_conj = jnp.conj(grad_eigenvalues)
    grad_eigenvectors_conj = jnp.conj(grad_eigenvectors)

    eigenvectors_H = matrix_adjoint(eigenvectors)
    dim = eigenvalues.shape[-1]
    eye_mask = jnp.eye(dim, dtype=bool)
    eye_mask = eye_mask.reshape((1,) * (eigenvalues.ndim - 1) + (dim, dim))

    # Then, the gradient is found by equation 4.77 of [2019 Boeddeker].
    rhs = (
        diag(grad_eigenvalues_conj)
        + jnp.conj(f_broadened) * (eigenvectors_H @ grad_eigenvectors_conj)
        - jnp.conj(f_broadened)
        * (eigenvectors_H @ eigenvectors)
        @ jnp.where(eye_mask, jnp.real(eigenvectors_H @ grad_eigenvectors_conj), 0.0)
    ) @ eigenvectors_H
    grad_matrix = jnp.linalg.solve(eigenvectors_H, rhs)

    # Take the conjugate of the gradient, reverting to the jax convention
    # where gradients are with respect to complex parameters.
    grad_matrix = jnp.conj(grad_matrix)

    # Return `grad_matrix`, and `None` for the gradient with respect to `eps`.
    return grad_matrix, None

def matrix_adjoint(x):
    """Computes the adjoint for a batch of matrices."""
    axes = tuple(range(x.ndim - 2)) + (x.ndim - 1, x.ndim - 2)
    return jnp.conj(jnp.transpose(x, axes=axes))

def diag(x):
    """A batch-compatible version of `numpy.diag`."""
    shape = x.shape + (x.shape[-1],)
    y = jnp.zeros(shape, x.dtype)
    i = jnp.arange(x.shape[-1])
    return y.at[..., i, i].set(x)

eig.defvjp(_eig_fwd, _eig_bwd)