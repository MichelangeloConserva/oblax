from dataclasses import dataclass
from itertools import count
from typing import NamedTuple, Tuple, Union, Callable

import chex
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
from acme.jax.networks import FeedForwardNetwork
from optax._src.base import GradientTransformation
from tqdm import trange, tqdm

from oblax.agents.bayesian_lin_reg_agent import (
    BeliefState as LinRegBeliefState,
    BayesianUnivariateLinearReg,
)
from oblax.agents.bayesian_lin_reg_agent import (
    SufficientStatistic as LinRegSufficientStatistic,
)
from oblax.agents.bayesian_multilin_reg_agent import (
    BeliefState as MVLinRegBeliefState,
    SufficientStatistic as MVLinRegSufficientStatistic,
    BayesianMultivariateLinearReg,
)

from seql.agents.base import Agent

tfd = tfp.distributions


class Info(NamedTuple):
    ...


@dataclass
class BeliefState:
    blr_belief_state: Union[LinRegBeliefState, MVLinRegBeliefState]
    neural_feature_network_params: chex.PyTreeDef
    neural_feature_network: Callable[[chex.PyTreeDef, chex.Array, bool], chex.Array]
    optimizer: GradientTransformation
    optimizer_state: tuple

    @property
    def is_multivariate(self):
        return type(self.blr_belief_state) == MVLinRegBeliefState


@dataclass
class SufficientStatistic:
    X: chex.Array
    Y: chex.Array
    n: int


class NeuralLinearAgent(Agent):
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
        # Obtain the data points from the replay buffer
        X, Y = suff_stats.X[: suff_stats.n], suff_stats.Y[: suff_stats.n]

        # Compute the feature maps
        phi = prior_belief.neural_feature_network(
            prior_belief.neural_feature_network_params, X, True
        )

        # Compute Bayesian linear regression
        XX, XY, YY, n = (
            phi.T @ phi,
            phi.T @ Y.squeeze(),
            (Y.T @ Y).squeeze(),
            suff_stats.n,
        )
        if prior_belief.is_multivariate:
            blr_ss = MVLinRegSufficientStatistic(XX, XY, YY, n)
            blr_posterior = BayesianMultivariateLinearReg.posterior_belief(
                prior_belief.blr_belief_state, blr_ss
            )
        else:
            blr_ss = LinRegSufficientStatistic(XX, XY, YY, n)
            blr_posterior = BayesianUnivariateLinearReg.posterior_belief(
                prior_belief.blr_belief_state, blr_ss
            )

        # Return updated posterior
        return BeliefState(
            blr_posterior,
            prior_belief.neural_feature_network_params,
            prior_belief.neural_feature_network,
            prior_belief.optimizer,
            prior_belief.optimizer_state,
        )

    def __init__(
        self,
        prior_belief: BeliefState,
        gradient_steps_per_update: int,
        input_dim: int,
        initial_capacity=int(1e3),
        is_classifier: bool = False,
    ):
        super(NeuralLinearAgent, self).__init__(is_classifier)
        self.input_dim = input_dim
        self.gradient_steps_per_update = gradient_steps_per_update
        self.capacity = initial_capacity

        if prior_belief.is_multivariate:
            self.m, self.d = prior_belief.blr_belief_state.M.shape
        else:
            self.d = len(prior_belief.blr_belief_state.mu)
            self.m = 1

        self.prior_belief = prior_belief
        self.suff_stats = SufficientStatistic(
            X=jnp.zeros((initial_capacity, self.input_dim)),
            Y=jnp.zeros((initial_capacity, self.m)),
            n=0,
        )
        self.cur_index = 0

        def model_fn(params, x):
            phi = prior_belief.neural_feature_network(params["net_params"], x, True)
            if self.prior_belief.is_multivariate:
                return phi @ params["mu_n"].T
            return phi @ params["mu_n"]

        self.model_fn = model_fn
        self._posterior_belief = None

    def _update_suff_stats(self, x: chex.Array, y: chex.Array):
        cur_n = len(x)

        while self.cur_index + cur_n > self.capacity:
            self.capacity *= 2

        if len(self.suff_stats.X) < self.capacity:
            n_add = self.capacity - len(self.suff_stats.X)
            self.suff_stats = SufficientStatistic(
                X=jnp.vstack((self.suff_stats.X, jnp.zeros((n_add, self.input_dim)))),
                Y=jnp.vstack((self.suff_stats.Y, jnp.zeros((n_add, self.m)))),
                n=self.suff_stats.n,
            )

        self.suff_stats = SufficientStatistic(
            self.suff_stats.X.at[self.cur_index : self.cur_index + cur_n].set(x),
            self.suff_stats.Y.at[self.cur_index : self.cur_index + cur_n].set(y),
            self.suff_stats.n + cur_n,
        )
        self.cur_index = self.suff_stats.n

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

        x = x.reshape(-1, self.input_dim)
        y = y.reshape(-1, self.m)

        self._update_suff_stats(x, y)

        if self.gradient_steps_per_update > 0:
            self.feature_map_learning(self.gradient_steps_per_update)

        if return_posterior:
            if belief is None:
                belief = self.prior_belief
            posterior_belief = NeuralLinearAgent.posterior_belief(
                belief, self.suff_stats
            )
            return posterior_belief, Info()

    def feature_map_learning(self, gradient_steps: int, belief: BeliefState = None):
        if belief is None:
            belief = self.prior_belief

        params = belief.neural_feature_network_params
        X, Y = (
            self.suff_stats.X[: self.cur_index],
            self.suff_stats.Y[: self.cur_index],
        )

        losses = []

        loop = tqdm(
            count() if gradient_steps == np.inf else range(gradient_steps),
            desc="Feature map learning",
        )
        for _ in loop:
            compute_loss = lambda params, x, y: optax.l2_loss(
                belief.neural_feature_network(params, x, False), y
            ).mean()
            loss, grads = jax.value_and_grad(compute_loss)(params, X, Y)
            updates, opt_state = belief.optimizer.update(grads, belief.optimizer_state)
            params = optax.apply_updates(params, updates)
            losses.append(loss)
            if len(losses) > 5:
                losses.pop(0)
            loop.set_postfix(dict(loss=f"{loss:.4f}"))
            if (
                len(losses) == 5
                and np.isclose([(loss - l) for l in losses[:-1]], 0.0).all()
            ):
                loop.close()
                break

        self.prior_belief = BeliefState(
            belief.blr_belief_state,
            params,
            belief.neural_feature_network,
            belief.optimizer,
            opt_state,
        )

    def sample_params(self, key: chex.PRNGKey, belief: BeliefState) -> chex.ArrayTree:
        """Sample parameters from the given belief."""

        if belief.is_multivariate:
            mu_n, sigma_squared = BayesianMultivariateLinearReg.jsample_params(
                key, belief.blr_belief_state
            )
        else:
            mu_n, sigma_squared = BayesianUnivariateLinearReg.jsample_params(
                key, belief.blr_belief_state
            )

        return dict(
            net_params=belief.neural_feature_network_params,
            mu_n=mu_n,
            sigma_squared=sigma_squared,
        )

    def predict_given_params_regression(self, params: chex.ArrayTree, x: chex.Array):
        # Regression with C outputs (independent)
        mu = self.model_fn(params, x)

        # n test examples, dimensionality of output
        ntest = len(mu)

        if self.prior_belief.is_multivariate:
            return tfd.MultivariateNormalFullCovariance(mu, params["sigma_squared"])
        return distrax.MultivariateNormalDiag(
            mu, params["sigma_squared"] * jnp.ones(ntest)
        )
