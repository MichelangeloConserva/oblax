import distrax
from flax.struct import dataclass
from typing import NamedTuple, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp


from seql.agents.base import Agent

tfd = tfp.distributions


class Info(NamedTuple):
    ...


# TODO: consider whether to unite the sufficient statistics with the belief state.


@dataclass
class BeliefState:
    mu: chex.Array
    Lambda: chex.Array
    a: float
    b: float


@dataclass
class SufficientStatistic:
    XX: chex.Array
    Xy: chex.Array
    yy: float
    n: int


class BayesianUnivariateLinearReg(Agent):
    @staticmethod
    def posterior_belief(
        prior_belief: BeliefState, suff_stats: SufficientStatistic
    ) -> BeliefState:
        """Compute the posterior belief of the Bayesian linear regression agent from a given prior and sufficient statistics.

        Args:
            prior_belief: the initial belief of the agent.
            suff_stats: the sufficient statistics of the observed data.

        Returns:
            The updated belief of the agent.
        """

        lambda_n = suff_stats.XX + prior_belief.Lambda
        mu_n = jnp.linalg.pinv(lambda_n) @ (
            suff_stats.Xy + prior_belief.Lambda @ prior_belief.mu
        )
        a_n = prior_belief.a + suff_stats.n / 2

        term1 = prior_belief.mu[None, :] @ prior_belief.Lambda @ prior_belief.mu
        term2 = mu_n[None, :] @ lambda_n @ mu_n
        b_n = float(prior_belief.b + 0.5 * (suff_stats.yy + term1 - term2))

        return BeliefState(mu_n, lambda_n, a_n, b_n)

    def __init__(self, prior_belief: BeliefState, is_classifier: bool = False):
        super(BayesianUnivariateLinearReg, self).__init__(is_classifier)

        self.d = d = len(prior_belief.mu)
        self.prior_belief = prior_belief
        self.suff_stats = SufficientStatistic(
            XX=jnp.zeros((d, d)), Xy=jnp.zeros((d,)), yy=0.0, n=0
        )

        self.model_fn = lambda params, x: x @ params
        self._posterior_belief = None

    def init_state(
        self, mu: chex.Array, Lambda: chex.Array, a: float, b: float
    ) -> BeliefState:
        """Return the object encoding the given belief."""
        return BeliefState(mu, Lambda, a, b)

    @staticmethod
    @jax.jit
    def _jupdate_suff_stats(
        suff_stats: SufficientStatistic, x: chex.Array, y: chex.Array
    ):
        return SufficientStatistic(
            suff_stats.XX + x.T @ x,
            suff_stats.Xy + (x.T @ y).squeeze(),
            suff_stats.yy + (y.T @ y).squeeze(),
            suff_stats.n + len(x),
        )

    def _update_suff_stats(self, x: chex.Array, y: chex.Array):
        self.suff_stats = self._jupdate_suff_stats(self.suff_stats, x, y)

    def update(
        self,
        key: chex.PRNGKey,
        belief: Union[BeliefState, None],
        x: chex.Array,
        y: chex.Array,
        return_posterior: bool = True,
    ) -> Union[Tuple[BeliefState, Info], None]:
        """Update the agent sufficient statistics given some observed data.

        Args:
            key: the random key. It is not used for this agent.
            belief: the starting belief of the agent. When set to None the agent will use its prior belief.
            x: the observed covariates.
            y: the observed variates.
            return_posterior: whether to return the posterior belief.

        Returns:
            If required it returns the posterior belief.
        """

        x = x.reshape(-1, self.d)
        y = y.reshape(-1, 1)

        self._update_suff_stats(x, y)

        if return_posterior:
            if belief is None:
                belief = self.prior_belief

            posterior_belief = BayesianUnivariateLinearReg.posterior_belief(
                belief, self.suff_stats
            )
            return posterior_belief, Info()

    @staticmethod
    @jax.jit
    def jsample_params(key: chex.PRNGKey, belief: BeliefState) -> chex.ArrayTree:
        sigma_squared = tfd.InverseGamma(belief.a, belief.b).sample(seed=key)

        theta = tfd.MultivariateNormalFullCovariance(
            belief.mu,
            sigma_squared * jnp.linalg.pinv(belief.Lambda)
            + jnp.eye(len(belief.Lambda)) * 1e-8,
        ).sample(seed=key)
        theta = theta.reshape(belief.mu.shape)
        return theta, sigma_squared

    def sample_params(self, key: chex.PRNGKey, belief: BeliefState) -> chex.ArrayTree:
        """Sample parameters from the given belief."""
        return self.jsample_params(key, belief)

    def predict_given_params_regression(self, params: chex.ArrayTree, x: chex.Array):
        """Predict the value of the given covariates."""
        mu_n, sigma_squared = params
        pred = self.model_fn(mu_n, x)

        # n test examples, dimensionality of output
        pred_dist = distrax.MultivariateNormalDiag(
            pred, sigma_squared * jnp.ones(len(pred))
        )

        return pred_dist
