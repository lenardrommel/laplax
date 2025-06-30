import gpjax as gpx
import jax
import jax.numpy as jnp
import matplotlib as mpl
from matplotlib import pyplot as plt

jax.config.update("jax_enable_x64", True)


class Prior:
    def __init__(self, kernel, key, data, dtype=jnp.float64, config=None):
        self.kernel = self._build_standard_prior("periodic")

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
                for i in range(1)
            ],
            axis=-1,
        )

        return prior_cross_covariance

    def _build_standard_prior(self, kernel_name):  # noqa: PLR6301
        if kernel_name == "periodic":
            kernel_per = gpx.kernels.Periodic(lengthscale=1.0, period=1.0)
            kernel_mat52 = gpx.kernels.Matern52(lengthscale=1.0, variance=0.25)
            kernel_mat12 = gpx.kernels.Matern12(lengthscale=0.25, variance=0.0001)

            kernel_time = gpx.kernels.ProductKernel([kernel_per, kernel_mat52])

            self.kernel = gpx.kernels.SumKernel([kernel_time, kernel_mat12])

        if kernel_name == "pureperiodic":
            self.kernel = gpx.kernels.Periodic(
                lengthscale=3.0,
                variance=1.0,
                period=6.0,
            )

        if kernel_name == "rbf":
            self.kernel = gpx.kernels.RBF()

        meanf = gpx.mean_functions.Zero()
        prior = gpx.gps.Prior(mean_function=meanf, kernel=self.kernel)
        return prior

    def _tune_hyperparameters(self, prior):
        posterior = prior * gpx.likelihoods.Gaussian(num_datapoints=self.data.n)
        opt_posterior, history = gpx.fit_scipy(
            model=posterior,
            # we use the negative mll as we are minimising
            objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
            train_data=self.data,
        )
        latent_dist = opt_posterior.predict(self.xtest, train_data=self.data)
        predictive_dist = opt_posterior.likelihood(latent_dist)

        predictive_mean = predictive_dist.mean
        predictive_std = jnp.sqrt(predictive_dist.variance)
        fig, ax = plt.subplots(figsize=(7.5, 2.5))
        x = self.data.X
        y = self.data.y
        xtest = self.xtest
        ytest = self.ytest
        cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        ax.plot(x, y, "x", label="Observations", color=cols[0], alpha=0.5)
        ax.fill_between(
            xtest.squeeze(),
            predictive_mean - 2 * predictive_std,
            predictive_mean + 2 * predictive_std,
            alpha=0.2,
            label="Two sigma",
            color=cols[1],
        )
        ax.plot(
            xtest,
            predictive_mean - 2 * predictive_std,
            linestyle="--",
            linewidth=1,
            color=cols[1],
        )
        ax.plot(
            xtest,
            predictive_mean + 2 * predictive_std,
            linestyle="--",
            linewidth=1,
            color=cols[1],
        )
        ax.plot(
            xtest,
            ytest,
            label="Latent function",
            color=cols[0],
            linestyle="--",
            linewidth=2,
        )
        ax.plot(xtest, predictive_mean, label="Predictive mean", color=cols[1])
        ax.legend(loc="center left", bbox_to_anchor=(0.975, 0.5))
        # plt.show()
        return opt_posterior.prior, history
