import numpy as onp
from jax.experimental import stax
from jax import numpy as np
from jax import grad, vmap
from scipy.optimize import root_scalar
from scipy.stats import gaussian_kde


class Loss():
    terminal_layer = None
    N_OUTPUTS = None

    def cumulative_hazard(self, params, t):
        # must override this or survival_function
        return -np.log(self.survival_function(params, t))

    def survival_function(self, params, t):
        # must override this or cumulative_hazard
        return np.exp(-self.cumulative_hazard(params, t))

    def hazard(self, params, t):
        return vmap(grad(self.cumulative_hazard))(params, t)

    def log_hazard(self, params, t):
        return np.log(self.hazard(params, t))

    def inform(self, **kwargs):
        raise NotImplementedError()



class GeneralizedGamma(Loss):

    N_OUTPUTS = 3

    def __init__(self, topology):
        self.terminal_layer = [stax.Dense(self.N_OUTPUTS)]

    def cumulative_hazard(self, params, t):
        pass

    def log_hazard(self, params, t):
        pass



class PiecewiseConstant(Loss):

    def __init__(self, breakpoints):
        self.N_OUTPUTS = len(breakpoints)
        self.breakpoints = breakpoints
        self.terminal_layer = [stax.Dense(self.N_OUTPUTS), stax.Exp]

    def cumulative_hazard(self, params, T):
        n = T.shape[0]
        T = T.reshape((n, 1))
        M = np.minimum(np.tile(self.breakpoints, (n, 1)), T)
        M = np.hstack([M[:, tuple([0])], np.diff(M, axis=1)])
        return (M * params).sum(1)

    def hazard(self, params, T):
        """
        The hazard is trivial for piecewise constant, but Numpy/Jax make it hard =(
        """
        n = T.shape[0]
        T = T.reshape((n, 1))
        tiles = np.tile(self.breakpoints[:-1], (n, 1))

        # watching issue #1142
        #ix = onp.searchsorted(self.breakpoints, T)
        ix = np.argmin(np.maximum(tiles, T) - tiles, 1)
        return params[np.arange(n), ix]


class NonParametric(PiecewiseConstant):
    """
    We create the concentration of breakpoints in proportional to the number of subjects that died at that time.
    """

    def __init__(self, n_breakpoints=None):
        self.n_breakpoints = n_breakpoints

    def inform(self, **kwargs):
        T = kwargs.pop("T")
        E = kwargs.pop("E")

        # first take a look at T, and create a KDE around the deaths
        breakpoints = self.create_breakpoints(T[E.astype(bool)])
        super(NonParametric, self).__init__(breakpoints)


    def create_breakpoints(self, observed_event_times):

        def solve_inverse_cdf_problem(p, dist, starting_point=0):
            f = lambda x: dist.integrate_box_1d(0, x) - p
            return root_scalar(f, x0=starting_point, fprime=dist).root


        n_obs = observed_event_times.shape[0]
        dist = gaussian_kde(observed_event_times)

        if self.n_breakpoints is None:
            n_breakpoints = int(n_obs / 15)
        else:
            n_breakpoints = self.n_breakpoints

        breakpoints = onp.empty(n_breakpoints+1)

        sol = 0
        for i, p in enumerate(np.linspace(0, 1, n_breakpoints+2)[1:-1]):
            # solve the following simple root problem:
            # cdf(x) = p
            sol = solve_inverse_cdf_problem(p, dist, starting_point=sol)
            breakpoints[i] = sol

        breakpoints[-1] = np.inf
        return breakpoints




