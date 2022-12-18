import chex
from matplotlib import pyplot as plt


def plot_vector(x: chex.Array, ax=None):
    """Plot the given vector on the given Axes."""

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(*x, alpha=0)
    ax.quiver(*[0] * len(x), *x, angles="xy", scale_units="xy", scale=1)
