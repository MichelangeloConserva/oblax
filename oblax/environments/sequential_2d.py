from typing import Callable

import chex
import jax
import haiku as hk
import jax.numpy as jnp

from oblax.environments.base import SequentialEnvironment, EnvParams


def vector2d_rotation(
    seed: int,
    N: int,
    deg_change: int,
    beta_norm: float,
    covariate_range,
) -> EnvParams:
    """

    Args:
        seed: a seed to control the randomness.
        N: the number of data points.
        deg_change: the degree for each rotation for the vector.
        beta_norm: the norm of the regression parameter vector.
        covariate_range: the range of the covariate.

    Returns:
        The variables defining the rotational environment.
    """
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

    return EnvParams(X, dict(beta=beta), rotation_update)


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
        beta_norm: the norm of the regression parameter vector. Default to 1.
        covariate_range: the range of the covariate. Default to (-1, 1).

    Returns:
        A SequentialEnvironment object corresponding to the given non-stationarities.
    """

    if batch_size is None:
        batch_size = N

    if observation_noise is None:
        observation_noise = lambda key, shape: jax.random.normal(key, shape)

    # Get the rotating
    env_params = vector2d_rotation(seed, N, deg_change, beta_norm, covariate_range)

    # Regression output function
    get_output = lambda key, X, p: X @ p["beta"] + observation_noise(key, X.shape[:-1])

    # Instantiate the sequential environment
    return SequentialEnvironment(seed, env_params, get_output, batch_size)


def logit_classification2d_rotation(
    seed: int,
    N: int,
    deg_change: int,
    batch_size: int = None,
    beta_norm: float = 1,
    covariate_range=(-7.5, 7.5),
) -> SequentialEnvironment:
    """Instantiate a non-stationary logit classification environment with rotating parameter.

    Args:
        seed: a seed to control the randomness.
        N: the number of data points.
        deg_change: the degree for each rotation for the vector.
        batch_size: the number of data points reveal to the agent at each time step. Default to all data points.
        beta_norm: the norm of the regression parameter vector.
        covariate_range: the range of the covariate.

    Returns:
        A SequentialEnvironment object corresponding to the given non-stationarities.
    """

    if batch_size is None:
        batch_size = N

    # Get the rotating
    env_params = vector2d_rotation(seed, N, deg_change, beta_norm, covariate_range)

    # Classification output function
    get_output = lambda key, X, p: jax.random.bernoulli(
        key, jax.nn.sigmoid(X @ p["beta"])
    ).astype(int)

    # Instantiate the sequential environment
    return SequentialEnvironment(seed, env_params, get_output, batch_size)
