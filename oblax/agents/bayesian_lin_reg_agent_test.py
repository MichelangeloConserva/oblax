import itertools

import chex
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow_probability.substrates.jax as tfp

from oblax.agents.bayesian_lin_reg_agent import BayesianUnivariateLinearReg, BeliefState
from oblax.environments.sequential_2d import linear_regression2d_rotation


tfd = tfp.distributions


class BayesLinRegTest(parameterized.TestCase):
    @parameterized.parameters([2, 10, 50])
    def test_init_state(self, d: int):

        a_0 = b_0 = 1.0
        lambda_0 = jnp.eye(d) * 0.1
        mu_0 = jnp.ones((d,)) * 0.5

        agent = BayesianUnivariateLinearReg(BeliefState(mu_0, lambda_0, a_0, b_0))

        prior_belief = agent.init_state(mu_0, lambda_0, a_0, b_0)
        prior_belief1 = agent.prior_belief

        chex.assert_type(prior_belief.a, float)
        chex.assert_type(prior_belief.b, float)
        chex.assert_type(prior_belief1.a, float)
        chex.assert_type(prior_belief1.b, float)

    @parameterized.parameters(itertools.product((4, 10), (10_000, 20_000), (0.1, 1.5)))
    def test_update(self, seed: int, N: int, sigma: float):
        obs_noise = lambda key, shape: jax.random.normal(key, shape) * sigma
        env = linear_regression2d_rotation(seed, N, 1, observation_noise=obs_noise)

        a_0 = b_0 = 1.0
        lambda_0 = jnp.eye(2) * 0.1
        mu_0 = jnp.ones((2,)) * 0.5
        agent = BayesianUnivariateLinearReg(BeliefState(mu_0, lambda_0, a_0, b_0))
        posterior_belief, _ = agent.update(None, None, *next(env))

        # Check if the posterior mean is close to the true parameter
        assert jnp.isclose(
            env.env_params["beta"], posterior_belief.mu, 1e-1, 1e-1
        ).all()

        # Check if the posterior variance mean is close to the true observation noise
        assert jnp.isclose(
            sigma ** 2,
            tfd.InverseGamma(posterior_belief.a, posterior_belief.b).mean(),
            1e-1,
            1e-1,
        )

    @parameterized.parameters(itertools.product((4, 10), (10, 25)))
    def test_sample_params(self, seed: int, d: int):

        a_0 = b_0 = 1.0
        lambda_0 = jnp.eye(d) * 0.1
        mu_0 = jnp.ones((d,)) * 0.5
        agent = BayesianUnivariateLinearReg(BeliefState(mu_0, lambda_0, a_0, b_0))

        belief = agent.init_state(mu_0, lambda_0, a_0, b_0)

        key = jax.random.PRNGKey(seed)

        theta, sigma_squared = agent.sample_params(key, belief)
        chex.assert_shape(theta, (d,))
        chex.assert_shape(sigma_squared, ())

    @parameterized.parameters(itertools.product((0,), (10,), (2,), (10,), (5,)))
    def test_posterior_predictive_sample(
        self,
        seed: int,
        ntrain: int,
        input_dim: int,
        nsamples_params: int,
        nsamples_output: int,
    ):
        output_dim = 1

        a_0 = b_0 = 1000.0
        lambda_0 = jnp.eye(2) * 0.01
        mu_0 = jnp.ones((2,)) * 1
        agent = BayesianUnivariateLinearReg(BeliefState(mu_0, lambda_0, a_0, b_0))

        key = jax.random.PRNGKey(seed)
        x_key, ppd_key = jax.random.split(key)

        x = jax.random.normal(x_key, shape=(ntrain, input_dim))
        samples = agent.posterior_predictive_sample(
            key, agent.prior_belief, x, nsamples_params, nsamples_output
        )
        chex.assert_shape(
            samples, (nsamples_params, ntrain, nsamples_output, output_dim)
        )

    @parameterized.parameters(itertools.product((0,), (5,), (2,), (10,)))
    def test_logprob_given_belief(
        self,
        seed: int,
        ntrain: int,
        input_dim: int,
        nsamples_params: int,
    ):
        a_0 = b_0 = 10.0
        lambda_0 = jnp.eye(input_dim) * 0.01
        mu_0 = jnp.ones((input_dim,)) * 1
        agent = BayesianUnivariateLinearReg(BeliefState(mu_0, lambda_0, a_0, b_0))

        key = jax.random.PRNGKey(seed)
        x_key, w_key, noise_key, logprob_key = jax.random.split(key, 4)

        x = jax.random.normal(x_key, shape=(ntrain, input_dim))
        w = jax.random.normal(w_key, shape=(input_dim, 1))
        y = x @ w + jax.random.normal(noise_key, (ntrain, 1))

        samples = agent.logprob_given_belief(
            logprob_key, agent.prior_belief, x, y, nsamples_params
        )
        chex.assert_shape(samples, (ntrain, 1))
        assert not jnp.any(jnp.isinf(samples))
        assert not jnp.any(jnp.isnan(samples))


if __name__ == "__main__":
    absltest.main()
