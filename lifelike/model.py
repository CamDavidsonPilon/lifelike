import numpy.random as npr
import numpy as onp
from jax import jit, grad, random, vmap
from jax import numpy as np
from jax.experimental import stax
from jax.experimental.optimizers import unpack_optimizer_state, pack_optimizer_state, l2_norm
from lifelike.utils import must_be_compiled_first

class Model:
    def __init__(self, topology):
        self.topology = topology
        self.is_compiled = False
        self.callbacks = []

    def _log_likelihood(self, params, T, E):
        n = T.shape[0]
        cum_hz = vmap(self.loss.cumulative_hazard)(params, T)
        log_hz = vmap(self.loss.log_hazard)(params, T)
        ll = 0
        ll = ll + (E * log_hz).sum()
        ll = ll + -(cum_hz).sum()
        return ll / n

    @must_be_compiled_first
    def fit(self, X, T, E, epochs=1, batch_size=32, callbacks=None, validation_split=0.0):
        rng = npr.RandomState(0)

        def data_stream(X, T, E, num_batches, num_train):
            while True:
              perm = rng.permutation(num_train)
              for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield X[batch_idx], T[batch_idx], E[batch_idx]

        def split_training_data(X, T, E, validation_split):
            n = X.shape[0]
            ix = onp.arange(n)
            rng.shuffle(ix)
            ix_train = ix[int(n * validation_split):]
            ix_test = ix[:int(n * validation_split)]

            return (X[ix_train], T[ix_train], E[ix_train]),\
                   (X[ix_test], T[ix_test], E[ix_test]),

        X, T, E = X.astype(float), T.astype(float), E.astype(float)
        (X_train, T_train, E_train), (X_test, T_test, E_test) = split_training_data(X, T, E, validation_split)

        num_train = X_train.shape[0]
        num_complete_batches, leftover = divmod(num_train, batch_size)
        num_batches = num_complete_batches + bool(leftover)
        batches = data_stream(X, T, E, num_batches, num_train)

        if callbacks is not None:
            self.callbacks.extend(callbacks)

        # some losses are created dynamically with knowledge of T
        self.loss.inform(T=T_train, E=E_train)
        self.topology.extend(self.loss.terminal_layer)

        init_random_params, self._predict = stax.serial(*self.topology)

        @jit
        def loss(weights, batch):
            X, T, E = batch
            params = self._predict(weights, X)
            return -self._log_likelihood(
                params, T, E
            )  + self.l2 * l2_norm(weights)

        @jit
        def update(i, opt_state, batch):
            weights = self.get_weights(opt_state)
            return self._opt_update(i, grad(loss)(weights, batch), opt_state)

        _, init_params = init_random_params(
            random.PRNGKey(0), (-1, X.shape[1])
        )  # why -1?
        self.opt_state = self._opt_init(init_params)

        # training loop
        epoch = 0
        continue_training = True
        while epoch < epochs and continue_training:
            for _ in range(num_batches):
                self.opt_state = update(epoch, self.opt_state, next(batches))

            for callback in self.callbacks:
                try:
                    callback(epoch=epoch,
                             model=self,
                             training_batch=(X_train, T_train, E_train),
                             testing_batch=(X_test, T_test, E_test),
                             loss=loss)
                except StopIteration:
                    continue_training = False
                    break

            epoch += 1

    def compile(self, optimizer=None, loss=None, optimizer_kwargs=None, l2=0.0):
        self.loss = loss
        self.is_compiled = True
        self.l2 = l2
        self.optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs
        self._opt_init, self._opt_update, self.get_weights = self.optimizer(
            **self._optimizer_kwargs
        )

    @must_be_compiled_first
    def evaluate(self, X, T, E):
        pass

    @must_be_compiled_first
    def predict_survival_function(self, x, t):
        weights = self.get_weights(self.opt_state)
        return vmap(self.loss.survival_function, in_axes=(None, 0))(
            self._predict(weights, x), t
        )

    @must_be_compiled_first
    def predict_cumulative_hazard(self, x, t):
        weights = self.get_weights(self.opt_state)
        return vmap(self.loss.cumulative_hazard, in_axes=(None, 0))(
            self._predict(weights, x), t
        )

    @must_be_compiled_first
    def predict_hazard(self, x, t):
        weights = self.get_weights(self.opt_state)
        return vmap(self.loss.hazard, in_axes=(None, 0))(
            self._predict(weights, x), t
        )

    def __getstate__(self):
        # This isn't scalable. I should remove this hardcoded stuff. Note
        # that _opt_init and _opt_update are not present due to PyCapsule pickling errors.
        d = {
            "opt_state": unpack_optimizer_state(self.opt_state),
            "get_weights": self.get_weights,
            "optimizer": self.optimizer,
            "l2": self.l2,
            "is_compiled": self.is_compiled,
            "callbacks": self.callbacks,
            "topology": self.topology,
            "loss": self.loss,
            "_optimizer_kwargs": self._optimizer_kwargs,
            "_predict": self._predict,
        }
        return d

    def __setstate__(self, d):
        d["_opt_init"], d["_opt_update"], d["get_weights"] = d["optimizer"](
            **d["_optimizer_kwargs"]
        )
        d["opt_state"] = pack_optimizer_state(d["opt_state"])
        self.__dict__ = d
        return
