import gpjax as gpx
import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)

KERNELS = {
    "RBF": gpx.kernels.RBF,
    "Matern12": gpx.kernels.Matern12,
    "Periodic": gpx.kernels.Periodic,
}


class Prior:
    def __init__(self, key, dataloader, config):
        self.key = key
        self.config = config
        self.kernel_name = config["kernel"]
        self.n_priors = 1

        self._build_standard_prior(config, dataloader)

    def __call__(self, x, jitter=1e-6):
        x = x.reshape(x.shape[0], -1)
        prior_mean = jnp.zeros((x.shape[0], 1))
        prior_cov = jnp.stack(
            [
                gpx.kernels.computations.DenseKernelComputation()
                .gram(self.kernel, x)
                .to_dense()
                + jitter * jnp.eye(x.shape[0])
                for i in range(1)
            ],
            axis=-1,
        )

        return prior_mean, prior_cov

    def cross_covariance(self, x1, x2, jitter=1e-6):
        x1 = x1.reshape(x1.shape[0], -1)
        x2 = x2.reshape(x2.shape[0], -1)
        prior_cross_covariance = jnp.stack(
            [
                gpx.kernels.computations.DenseKernelComputation()
                .cross_covariance(self.kernel, x1, x2)
                .to_dense()
                + jitter * jnp.eye(x1.shape[0])
                for i in range(self.n_priors)
            ],
            axis=-1,
        )

        return prior_cross_covariance

    def marginal_variance(self, x):
        x = x.reshape(x.shape[0], 1, -1)
        prior_var = jnp.stack(
            [
                jax.vmap(
                    lambda x: gpx.kernels.computations.DenseKernelComputation().gram(
                        self.kernel, x
                    )
                )(x)
                .to_dense()
                .reshape(-1)
                for i in range(self.n_priors)
            ],
            axis=-1,
        )

        return prior_var

    def covariance_trace(self, x):
        x = x.reshape(x.shape[0], 1, -1)

        prior_trace = jnp.stack(
            [
                jax.vmap(
                    lambda _x: gpx.kernels.computations.DenseKernelComputation()
                    .gram(self.kernel, _x)
                    .to_dense()
                )(x).sum()
                for i in range(self.n_priors)
            ],
            axis=-1,
        )

        return prior_trace

    def _build_standard_prior(self, kernel_name):  # noqa: PLR6301
        if kernel_name == "periodic":
            k1 = gpx.kernels.Periodic()
            k2 = gpx.kernels.Matern52()
            kt = gpx.kernels.ProductKernel(kernels=[k1, k2])
            ks = gpx.kernels.Matern12()
            self.kernel = gpx.kernels.SumKernel(kernels=[kt, ks])

        if kernel_name == "pureperiodic":
            self.kernel = gpx.kernels.Periodic(lengthscale=0.01, period=jnp.pi * 2.0)

        if kernel_name == "rbf":
            self.kernel = gpx.kernels.RBF()

        meanf = gpx.mean_functions.Zero()
        prior = gpx.gps.Prior(mean_function=meanf, kernel=self.kernel)
        return prior
