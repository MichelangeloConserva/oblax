import itertools

import chex
import jax
import jax.numpy as jnp
import optax

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow_probability.substrates.jax as tfp
from acme.jax.networks import FeedForwardNetwork

from oblax.agents.bayesian_lin_reg_agent import BayesianUnivariateLinearReg
from oblax.agents.neural_linear_agent import NeuralLinearAgent, BeliefState
from oblax.environments.sequential_2d import linear_regression2d_rotation
from oblax.agents.bayesian_lin_reg_agent import BeliefState as LinRegBeliefState
from oblax.agents.bayesian_multilin_reg_agent import BeliefState as MRegBeliefState
import haiku as hk

tfd = tfp.distributions

jax.config.update("jax_platform_name", "cpu")

optimizer = optax.adam(1e-3)


def get_net_f(d, m):
    def net(x: chex.Array, get_phi: bool):
        phi_net = hk.Sequential(
            [
                hk.Linear(64),
                jax.nn.leaky_relu,
                hk.Linear(64),
                jax.nn.leaky_relu,
                hk.Linear(d),
                # jax.nn.leaky_relu,
            ]
        )
        W = hk.Linear(m, with_bias=False, name="W")
        phi = phi_net(x)
        return phi if get_phi else W(phi)

    return net


def get_agent(input_dim, D, m, gradient_steps_per_update):
    key = jax.random.PRNGKey(42)

    net = get_net_f(D, m)
    network = hk.without_apply_rng(hk.transform(net))
    network = FeedForwardNetwork(
        network.init, jax.jit(network.apply, static_argnums=(2,))
    )
    init_params = network.init(key, jnp.zeros(input_dim), False)

    if m == 1:
        blr_prior = LinRegBeliefState(
            mu=jnp.ones((D,)) * 1,
            Lambda=jnp.eye(D) * 0.01,
            a=10.0,
            b=10.0,
        )
    else:
        blr_prior = MRegBeliefState(jnp.ones((m, D)), jnp.eye(D), m + 1.0, jnp.eye(m))

    prior_belief = BeliefState(
        blr_prior,
        init_params,
        network.apply,
        optimizer,
        optimizer.init(init_params),
    )

    return (
        NeuralLinearAgent(
            prior_belief,
            input_dim=input_dim,
            gradient_steps_per_update=gradient_steps_per_update,
        ),
        prior_belief,
    )


class NeuralLinearTest(parameterized.TestCase):
    @parameterized.parameters(itertools.product([2, 10, 50], [64, 32], [10], [1, 5]))
    def test_init_state(
        self, input_dim: int, d: int, gradient_steps_per_update: int, m: int
    ):
        agent, prior_belief = get_agent(input_dim, d, m, gradient_steps_per_update)

        prior_belief_a = agent.prior_belief

        if prior_belief.is_multivariate:
            chex.assert_shape(prior_belief.blr_belief_state.M, (m, d))
            chex.assert_shape(prior_belief.blr_belief_state.V, (d, d))
            chex.assert_shape(prior_belief.blr_belief_state.Psi, (m, m))
            chex.assert_type(prior_belief.blr_belief_state.nu, float)

            chex.assert_shape(prior_belief_a.blr_belief_state.M, (m, d))
            chex.assert_shape(prior_belief_a.blr_belief_state.V, (d, d))
            chex.assert_shape(prior_belief_a.blr_belief_state.Psi, (m, m))
            chex.assert_type(prior_belief_a.blr_belief_state.nu, float)
        else:
            chex.assert_type(prior_belief.blr_belief_state.a, float)
            chex.assert_type(prior_belief.blr_belief_state.b, float)
            chex.assert_type(prior_belief_a.blr_belief_state.a, float)
            chex.assert_type(prior_belief_a.blr_belief_state.b, float)

    @parameterized.parameters(itertools.product((4, 10), (2000, 1000), [32, 64]))
    def test_update(self, seed: int, N: int, d: int):
        obs_noise = lambda key, shape: jax.random.normal(key, shape) * 0.00001
        env = linear_regression2d_rotation(seed, N, 1, observation_noise=obs_noise)

        agent, prior_belief = get_agent(2, d, 1, 500)

        x, y = next(env)
        posterior_belief, _ = agent.update(None, None, x, y)

        # Check nn predictions
        net, params = (
            posterior_belief.neural_feature_network,
            posterior_belief.neural_feature_network_params,
        )
        y_hat = net(params, x, False)
        assert ((y - y_hat) ** 2).mean() < 1

        # Check posterior predictions
        phi = net(params, x, True)
        if prior_belief.is_multivariate:
            y_hat = phi @ posterior_belief.blr_belief_state.M.T
        else:
            y_hat = phi @ posterior_belief.blr_belief_state.mu.T
        assert ((y - y_hat) ** 2).mean() < 1

    @parameterized.parameters(itertools.product((4, 10), (10, 25), (10, 20), (3, 40)))
    def test_sample_params(self, seed: int, input_dim: int, d: int, m: int):

        agent, prior_belief = get_agent(input_dim, d, m, 250)
        belief = agent.prior_belief

        key = jax.random.PRNGKey(seed)
        params = agent.sample_params(key, belief)

        if prior_belief.is_multivariate:
            chex.assert_shape(params["mu_n"], (m, d))
            chex.assert_shape(params["sigma_squared"], (m, m))
        else:
            chex.assert_shape(params["mu_n"], (d,))
            chex.assert_shape(params["sigma_squared"], ())

    @parameterized.parameters(
        itertools.product((0,), (10,), (2,), (10,), (5,), (10,), (1, 3))
    )
    def test_posterior_predictive_sample(
        self,
        seed: int,
        ntrain: int,
        input_dim: int,
        nsamples_params: int,
        nsamples_output: int,
        d: int,
        m: int,
    ):
        agent, prior_belief = get_agent(input_dim, d, m, 250)

        key = jax.random.PRNGKey(seed)
        x_key, ppd_key = jax.random.split(key)

        x = jax.random.normal(x_key, shape=(ntrain, input_dim))
        samples = agent.posterior_predictive_sample(
            key, agent.prior_belief, x, nsamples_params, nsamples_output
        )
        chex.assert_shape(samples, (nsamples_params, ntrain, nsamples_output, m))

    @parameterized.parameters(itertools.product((0,), (5,), (2,), (10,), (10,), (3, 1)))
    def test_logprob_given_belief(
        self,
        seed: int,
        ntrain: int,
        input_dim: int,
        nsamples_params: int,
        d: int,
        m: int,
    ):
        agent, prior_belief = get_agent(input_dim, d, m, 5)

        key = jax.random.PRNGKey(seed)
        x_key, w_key, noise_key, logprob_key = jax.random.split(key, 4)

        x = jax.random.normal(x_key, shape=(ntrain, input_dim))
        w = jax.random.normal(w_key, shape=(input_dim, m))
        y = x @ w + jax.random.normal(noise_key, (ntrain, m))

        posterior, _ = agent.update(key, None, x, y, return_posterior=True)

        samples = agent.logprob_given_belief(
            logprob_key, posterior, x, y, nsamples_params
        )
        chex.assert_shape(samples, (ntrain, 1))
        assert not jnp.any(jnp.isinf(samples))
        assert not jnp.any(jnp.isnan(samples))


if __name__ == "__main__":
    absltest.main()
