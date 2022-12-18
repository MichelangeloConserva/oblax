from typing import Callable

import chex
import jax
import haiku as hk
import jax.numpy as jnp
from matplotlib import pyplot as plt

from oblax.environments.base import SequentialEnvironment
from oblax.utils import plot_vector


def linear_regression2d_rotation(
    seed: int,
    N: int,
    deg_change: int,
    batch_size: int = None,
    observation_noise: Callable[[chex.PRNGKey, chex.Shape], chex.Array] = None,
    beta_norm: float = 1,
    covariate_range=(-1, 1),
) -> SequentialEnvironment:
    """Instantiate a non-stationary linear regression environment with rotating parameter.

    Args:
        seed: a seed to control the randomness.
        N: the number of data points.
        deg_change: the degree for each rotation for the vector.
        batch_size: the number of data points reveal to the agent at each time step. Default to all data points.
        observation_noise: the observation noise. Default to Gaussian noise with unitary scale.
        beta_norm: the norm of the regression parameter vector.
        covariate_range: the range of the covariate.

    Returns:
        A SequentialEnvironment object corresponding to the given non-stationarities.
    """

    if batch_size is None:
        batch_size = N

    if observation_noise is None:
        observation_noise = lambda key, shape: jax.random.normal(key, shape)

    rng = hk.PRNGSequence(seed)

    # Instantiate the covariates
    X = jax.random.uniform(next(rng), (N, 2), float, *covariate_range)

    # Parameters of the non-stationary environment
    beta = jax.random.normal(next(rng), (2,))
    beta = beta_norm * beta / jnp.linalg.norm(beta)

    # Rotation matrix function
    R = lambda theta: jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
    )

    # Update rule for non-stationary environment parameters
    rotation_update = lambda p: R(jnp.deg2rad(deg_change)) @ p

    # Regression output
    get_output = lambda key, X, p: X @ p["beta"] + observation_noise(key, X.shape[:-1])

    # Instantiate the sequential environment
    return SequentialEnvironment(
        seed, dict(beta=beta), X, rotation_update, get_output, batch_size
    )


def linear_regression2d_anim_update(num, fig, env):
    """Produce the figure update to display animation visualization of the linear_regression2d non-stationary environment."""
    fig.clear()

    Y_shown = next(env)

    x1, x2 = env.covariates.T
    xx, yy = jnp.meshgrid(
        jnp.linspace(x1.min(), x1.max(), 10), jnp.linspace(x2.min(), x2.max(), 10)
    )
    X = jnp.concatenate((xx[..., None], yy[..., None]), -1)

    Y = env.get_current_full_output()

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(xx, yy, X @ env.env_params["beta"], alpha=0.3)
    ax1.scatter(*env.covariates.T, Y, color="tab:green")
    ax1.set_zlim(Y.min(), Y.max())
    ax1.set_xlabel("$X_1$")
    ax1.set_ylabel("$X_2$")
    ax1.set_zlabel("$Y$")
    ax1.set_title("Full data set")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.scatter(*env.covariates[env.indices].T, Y_shown, color="tab:blue")
    ax2.set_xlim(x1.min(), x1.max())
    ax2.set_ylim(x2.min(), x2.max())
    ax2.set_zlim(Y.min(), Y.max())
    ax2.set_xlabel("$X_1$")
    ax2.set_ylabel("$X_2$")
    ax2.set_zlabel("$Y$")
    ax2.set_title("Data points shown to the agent")

    n = jnp.linalg.norm(env.env_params["beta"]) * 1.1
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("Beta vector")
    plot_vector(env.env_params["beta"], ax3)
    ax3.set_ylim(-n, n)
    ax3.set_xlim(-n, n)

    plt.tight_layout(w_pad=5)
