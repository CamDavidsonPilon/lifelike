from jax import jit, grad, random, vmap
from jax.experimental import stax
from lifelike.callbacks import Logger

class Model:

    def __init__(self, topology):
        self.topology = topology
        self.compiled = False
        self.callbacks = []

    def _log_likelihood(self, params, T, E):
        n = params.shape[0]
        cum_hz = vmap(self.loss.cumulative_hazard)(params, T)
        log_hz = vmap(self.loss.log_hazard)(params, T)
        ll = 0
        ll = ll + (E * log_hz).sum()
        ll = ll + -(cum_hz).sum()
        return ll / n

    #@must_be_compiled_first
    def fit(self, X, T, E, epochs=1000, batch_size=32, callbacks=None):

        X, T, E = X.astype(float), T.astype(float), E.astype(float)

        if callbacks is not None:
            self.callbacks.extend(callbacks)

        if not self.compiled:
            raise ValueError("Must run `compile` first.")

        # some losses are created dynamically with knowledge of T
        self.loss.inform(T=T, E=E)
        self.topology.extend(self.loss.terminal_layer)

        init_random_params, predict = stax.serial(*self.topology)
        opt_init, opt_update, get_weights = self.optimizer(**self._optimizer_kwargs)

        @jit
        def loss(weights, batch):
            X, T, E = batch
            params = predict(weights, X)
            return -self._log_likelihood(params, T, E)# + L2_reg * optimizers.l2_norm(weights)

        @jit
        def update(i, opt_state, batch):
           weights = get_weights(opt_state)
           return opt_update(i, grad(loss)(weights, batch), opt_state)


        _, init_params = init_random_params(random.PRNGKey(0), (-1, X.shape[1])) # why -1?
        opt_state = opt_init(init_params)

        for epoch in range(epochs):
            opt_state = update(epoch, opt_state, (X, T, E))

            for callback in self.callbacks:
                callback(epoch,
                    opt_state=opt_state,
                    get_weights=get_weights,
                    batch=(X, T, E),
                    loss_function=loss,
                    predict=predict,
                    loss=self.loss)


    def compile(self, optimizer=None, loss=None, optimizer_kwargs=None):
        self.loss = loss
        self.compiled = True
        self.optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs


    def evaluate(self, X, T, E):
        pass



    def predict(self, X):
        pass