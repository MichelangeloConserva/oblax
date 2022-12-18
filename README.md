# OBLax

OBLax is a collection of online Bayesian learning algorithms implemented in jax.

If you are interested in contributing you can check the project [Kanban board](https://github.com/users/MichelangeloConserva/projects/1).


## Non-stationary environment examples

### Linear regression 2d with non-stationary rotating parameter vector

For a univariate linear regression problem in the form,
$$
Y = X \boldsymbol \beta_t + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2).
$$

At each time step, the parameter vector $\boldsymbol \beta_t$ is rotated as,
$$\boldsymbol \beta_{t + 1} = R(\theta) \boldsymbol \beta_t,$$
for a fixed value $\theta$.

The rotation matrix is defined as,

$$
R(\theta) =
\begin{bmatrix}
cos(\theta) & -\sin(\theta) \\
cos(\theta) & \sin(\theta)
\end{bmatrix} 
$$

At each time step, the agent is revealed only a subset of the entire data set.

A visualization of how the environment changes for a full rotation of the paramter is presented below.

<img src="images/regression2d_rotation.gif" width=75%>

The code to reproduce the gif can be found in the "regression2d_rotation" notebook in the examples folder.
