from copy import deepcopy
from typing import Callable, TYPE_CHECKING

import chex
import jax
import haiku as hk
from flax import struct

if TYPE_CHECKING:
    pass


@struct.dataclass
class EnvParams:
    covariates: chex.Array
    initial_env_params: chex.PyTreeDef
    param_update: Callable


class SequentialEnvironment:
    def __init__(
        self,
        seed: int,
        env_params: "EnvParams",
        output_f: Callable[[chex.PRNGKey, chex.Array, chex.PyTreeDef], chex.Array],
        batch_size: int,
    ):
        self.initial_env_params = env_params.initial_env_params
        self.param_update = env_params.param_update
        self.output_f = output_f
        self.covariates = env_params.covariates
        self.batch_size = batch_size

        self.indices = None
        self.N = len(env_params.covariates)
        self.env_params = deepcopy(env_params.initial_env_params)
        self.rng = hk.PRNGSequence(seed)
        self.cur_key = next(self.rng)

        assert self.N >= batch_size

    def update_randomness(self):
        self.cur_key = next(self.rng)

    def update(self):
        self.env_params = jax.tree_util.tree_map(self.param_update, self.env_params)

    def get_current_full_output(self, X: chex.Array = None):
        if X is None:
            X = self.covariates
        return self.output_f(self.cur_key, X, self.env_params)

    def __next__(self):
        # Update environment parameters
        self.update()

        # Sample indices of covariates that will be shown to the agent
        self.indices = jax.random.choice(
            next(self.rng), self.N, (self.batch_size,), False
        )

        # Compute and return the output for the given indices
        return self.output_f(
            self.cur_key, self.covariates[self.indices], self.env_params
        )
