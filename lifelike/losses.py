from jax.experimental import stax
from jax import numpy as np
from jax import grad, vmap


class Loss():
    terminal_layer = None

    def cumulative_hazard(params, t):
        # must override this or survival_function
        return -np.log(self.survival_function(params, t))

    def survival_function(params, t):
        # must override this or cumulative_hazard
        return np.exp(-self.cumulative_hazard(params, t))

    def hazard(params, t):
        return vmap(grad(self.cumulative_hazard))(params, t)

    def log_hazard(params, t):
        return np.log(self.hazard(params, t))



class GeneralizedGamma(Loss):

    N_OUTPUTS = 3

    def __init__(self, topology):
        self.terminal_layer = [stax.Dense(self.N_OUTPUTS)]

    def cumulative_hazard(params, t):
        pass



class PiecewiseConstant(Loss):

    def __init__(self, breakpoints):
        self.N_OUTPUTS = len(breakpoints)
        self.terminal_layer = [stax.Dense(self.N_OUTPUTS), Exp]

    def _cumulative_hazard(params, T):
        n = T.shape[0]
        T = T.reshape((n, 1))
        M = np.minimum(np.tile(BREAKPOINTS, (n, 1)), T)
        M = np.hstack([M[:, tuple([0])], np.diff(M, axis=1)])
        return (M * params).sum(1)

    def _hazard(params, T):
        """
        The hazard is trivial for piecewise constant, but Numpy/Jax make it hard =(
        """
        n = T.shape[0]
        T = T.reshape((n, 1))
        tiles = np.tile(BREAKPOINTS[:-1], (n, 1))

        # watching issue #1142
        #ix = onp.searchsorted(BREAKPOINTS, T)
        ix = np.argmin(np.maximum(tiles, T) - tiles, 1)
        return params[np.arange(n), ix]


class NonParametric(PiecewiseConstant):
    """
    We create concentration of breakpoints proportional the number of subjects still alive.
    """

    def __init__(self):
        super(self, NonParametric).__init__(breakpoints)


