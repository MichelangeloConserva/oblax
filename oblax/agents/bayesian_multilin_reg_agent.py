from flax.struct import dataclass
from typing import NamedTuple, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

import dynamax
from dynamax.utils.distributions import MatrixNormalInverseWishart
from dynamax.utils.utils import psd_solve

from seql.agents.base import Agent

tfd = tfp.distributions


class Info(NamedTuple):
    ...


# TODO: consider whether to unite the sufficient statistics with the belief state.


@dataclass
class BeliefState:
    M: chex.Array
    V: chex.Array
    nu: float
    Psi: chex.Array


@dataclass
class SufficientStatistic:
    XX: chex.Array
    XY: chex.Array
    YY: chex.Array
    n: int


class BayesianMultivariateLinearReg(Agent):
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

        Sxx = prior_belief.V + suff_stats.XX
        Sxy = suff_stats.XY + prior_belief.V @ prior_belief.M.T
        Syy = suff_stats.YY + prior_belief.M @ prior_belief.V @ prior_belief.M.T
        M_pos = psd_solve(Sxx, Sxy).T
        V_pos = Sxx
        nu_pos = prior_belief.nu + suff_stats.n
        Psi_pos = prior_belief.Psi + Syy - M_pos @ Sxy

        return BeliefState(M_pos, V_pos, nu_pos, Psi_pos)

    def __init__(self, prior_belief: BeliefState, is_classifier: bool = False):
        super(BayesianMultivariateLinearReg, self).__init__(is_classifier)

        self.m, self.d = prior_belief.M.shape

        self.prior_belief = prior_belief
        self.suff_stats = SufficientStatistic(
            XX=jnp.zeros((self.d, self.d)),
            XY=jnp.zeros((self.d, self.m)),
            YY=jnp.zeros((self.m, self.m)),
            n=0,
        )

        self.model_fn = lambda params, x: x @ params.T
        self._posterior_belief = None

    def init_state(
        self, M: chex.Array, V: chex.Array, nu: float, Psi: chex.Array
    ) -> BeliefState:
        """Return the object encoding the given belief."""
        return BeliefState(M, V, nu, Psi)

    @staticmethod
    @jax.jit
    def _jupdate_suff_stats(
        suff_stats: SufficientStatistic, x: chex.Array, y: chex.Array
    ):
        return SufficientStatistic(
            suff_stats.XX + x.T @ x,
            suff_stats.XY + (x.T @ y).squeeze(),
            suff_stats.YY + y.T @ y,
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

        if belief is None:
            belief = self.prior_belief

        x = x.reshape(-1, self.d)
        y = y.reshape(-1, self.m)

        self._update_suff_stats(x, y)

        if return_posterior:
            posterior_belief = BayesianMultivariateLinearReg.posterior_belief(
                belief, self.suff_stats
            )
            return posterior_belief, Info()

    @staticmethod
    @jax.jit
    def jsample_params(key: chex.PRNGKey, belief: BeliefState) -> chex.ArrayTree:
        mniw = MatrixNormalInverseWishart(belief.M, belief.V, belief.nu, belief.Psi)
        Sigma_samples, Matrix_samples = mniw.sample(seed=key)
        return Matrix_samples, Sigma_samples

    def sample_params(self, key: chex.PRNGKey, belief: BeliefState) -> chex.ArrayTree:
        """Sample parameters from the given belief."""
        return self.jsample_params(key, belief)

    def predict_given_params_regression(self, params: chex.ArrayTree, x: chex.Array):
        """Predict the value of the given covariates."""
        Matrix_samples, Sigma_samples = params
        pred = self.model_fn(Matrix_samples, x)

        # n test examples, dimensionality of output
        # pred_dist = distrax.MultivariateNormalDiag(loc=pred, covariance=sigma_squared * jnp.eye(len(pred))) # the output shape is (len(pred), len(pred)) for some reason here
        pred_dist = tfd.MultivariateNormalFullCovariance(pred, Sigma_samples)

        return pred_dist
