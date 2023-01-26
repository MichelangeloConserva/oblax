import itertools

import chex
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow_probability.substrates.jax as tfp
from dynamax.utils.distributions import InverseWishart

from oblax.agents.bayesian_multilin_reg_agent import (
    BayesianMultivariateLinearReg,
    BeliefState,
)
from oblax.environments.sequential_2d import linear_regression2d_rotation


tfd = tfp.distributions


class BayesMultiLinRegTest(parameterized.TestCase):
    @parameterized.parameters(itertools.product([2, 10, 50], [2, 10, 50]))
    def test_init_state(self, d: int, m: int):

        loc = jnp.ones((m, d))
        col_precision = jnp.eye(d)
        df = m + 1.0
        scale = jnp.eye(m)
        agent = BayesianMultivariateLinearReg(
            BeliefState(loc, col_precision, df, scale)
        )

        prior_belief = agent.init_state(loc, col_precision, df, scale)
        prior_belief1 = agent.prior_belief

        chex.assert_shape(prior_belief.M, (m, d))
        chex.assert_shape(prior_belief.V, (d, d))
        chex.assert_shape(prior_belief.Psi, (m, m))
        chex.assert_type(prior_belief.nu, float)

        chex.assert_shape(prior_belief1.M, (m, d))
        chex.assert_shape(prior_belief1.V, (d, d))
        chex.assert_shape(prior_belief1.Psi, (m, m))
        chex.assert_type(prior_belief1.nu, float)

    @parameterized.parameters(
        itertools.product((4, 10), (2000, 1000), [2, 10], [2, 10])
    )
    def test_update(self, seed: int, N: int, d: int, m: int):

        x_key, w_key, noise_key, logprob_key = jax.random.split(
            jax.random.PRNGKey(seed), 4
        )
        x = jax.random.normal(x_key, shape=(N, d))
        w = jax.random.normal(w_key, shape=(d, m))
        y = x @ w + jax.random.normal(noise_key, (N, m))

        loc = jnp.ones((m, d))
        col_precision = jnp.eye(d)
        df = m + 1.0
        scale = jnp.eye(m)
        agent = BayesianMultivariateLinearReg(
            BeliefState(loc, col_precision, df, scale)
        )

        posterior_belief, _ = agent.update(None, None, x, y)

        # Check if the posterior mean is close to the true parameter
        assert jnp.isclose(w, posterior_belief.M.T, 1e-1, 1e-1).all()

        # Check if the posterior variance mean is close to the true observation noise
        assert jnp.isclose(
            jnp.eye(m),
            InverseWishart(posterior_belief.nu, posterior_belief.Psi).mean(),
            1e-1,
            1e-1,
        ).all()

    @parameterized.parameters(itertools.product((4, 10), (10, 25), (10, 25)))
    def test_sample_params(self, seed: int, d: int, m: int):
        loc = jnp.ones((m, d))
        col_precision = jnp.eye(d)
        df = m + 1.0
        scale = jnp.eye(m)
        agent = BayesianMultivariateLinearReg(
            BeliefState(loc, col_precision, df, scale)
        )
        key = jax.random.PRNGKey(seed)

        Matrix_samples, Sigma_samples = agent.sample_params(key, agent.prior_belief)
        chex.assert_shape(Matrix_samples, (m, d))
        chex.assert_shape(Sigma_samples, (m, m))

    @parameterized.parameters(itertools.product((0,), (10,), (2,), (10,), (5,), (4,)))
    def test_posterior_predictive_sample(
        self,
        seed: int,
        ntrain: int,
        input_dim: int,
        nsamples_params: int,
        nsamples_output: int,
        output_dim: int,
    ):
        m, d, = (
            output_dim,
            input_dim,
        )

        loc = jnp.ones((m, d))
        col_precision = jnp.eye(d)
        df = m + 1.0
        scale = jnp.eye(m)
        agent = BayesianMultivariateLinearReg(
            BeliefState(loc, col_precision, df, scale)
        )

        key = jax.random.PRNGKey(seed)
        x_key, ppd_key = jax.random.split(key)

        x = jax.random.normal(x_key, shape=(ntrain, input_dim))
        samples = agent.posterior_predictive_sample(
            key, agent.prior_belief, x, nsamples_params, nsamples_output
        )
        chex.assert_shape(
            samples, (nsamples_params, ntrain, nsamples_output, output_dim)
        )

    @parameterized.parameters(itertools.product((0,), (5,), (2,), (10,), (5,)))
    def test_logprob_given_belief(
        self,
        seed: int,
        ntrain: int,
        input_dim: int,
        nsamples_params: int,
        output_dim: int,
    ):
        m, d, = (
            output_dim,
            input_dim,
        )

        loc = jnp.ones((m, d))
        col_precision = jnp.eye(d)
        df = m + 1.0
        scale = jnp.eye(m)
        agent = BayesianMultivariateLinearReg(
            BeliefState(loc, col_precision, df, scale)
        )

        key = jax.random.PRNGKey(seed)
        x_key, w_key, noise_key, logprob_key = jax.random.split(key, 4)

        x = jax.random.normal(x_key, shape=(ntrain, input_dim))
        w = jax.random.normal(w_key, shape=(input_dim, output_dim))
        y = x @ w + jax.random.normal(noise_key, (ntrain, output_dim))

        samples = agent.logprob_given_belief(
            logprob_key, agent.prior_belief, x, y, nsamples_params
        )
        chex.assert_shape(samples, (ntrain, 1))
        assert not jnp.any(jnp.isinf(samples))
        assert not jnp.any(jnp.isnan(samples))


if __name__ == "__main__":
    absltest.main()
