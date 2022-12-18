import chex
import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt

from oblax.utils import plot_vector


def _plot_env_vector(x: chex.Array, ax, title=None):
    n = jnp.linalg.norm(x) * 1.1
    if title:
        ax.set_title(title)
    plot_vector(x, ax)
    ax.set_ylim(-n, n)
    ax.set_xlim(-n, n)


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

    _plot_env_vector(env.env_params["beta"], fig.add_subplot(1, 3, 3), "Beta vector")

    plt.tight_layout(w_pad=5)


def logit_classification2d_anim_update(num, fig, env):
    """Produce the figure update to display animation visualization of the classification_regression2d non-stationary environment."""
    fig.clear()

    Y_shown = next(env)

    x1, x2 = env.covariates.T
    xx, yy = jnp.meshgrid(
        jnp.linspace(x1.min(), x1.max(), 100), jnp.linspace(x2.min(), x2.max(), 100)
    )
    X = jnp.concatenate((xx[..., None], yy[..., None]), -1)

    Y = env.get_current_full_output()

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.contourf(
        xx,
        yy,
        (jax.nn.sigmoid(X @ env.env_params["beta"]) > 0.5).astype(int),
        cmap=plt.cm.coolwarm,
    )
    ax1.scatter(*env.covariates.T, c=Y, cmap=plt.cm.seismic)
    ax1.set_xlabel("$X_1$")
    ax1.set_ylabel("$X_2$")
    ax1.set_title("Full data set")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.contourf(
        xx,
        yy,
        (jax.nn.sigmoid(X @ env.env_params["beta"]) > 0.5).astype(int),
        cmap=plt.cm.coolwarm,
    )
    ax2.scatter(*env.covariates[env.indices].T, c=Y_shown, cmap=plt.cm.seismic)
    ax2.set_xlim(x1.min(), x1.max())
    ax2.set_ylim(x2.min(), x2.max())
    ax2.set_xlabel("$X_1$")
    ax2.set_ylabel("$X_2$")
    ax2.set_title("Data points shown to the agent")

    _plot_env_vector(env.env_params["beta"], fig.add_subplot(1, 3, 3), "Beta vector")

    plt.tight_layout(w_pad=5)
