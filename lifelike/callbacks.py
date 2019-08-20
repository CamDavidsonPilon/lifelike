from pathlib import Path
import time

import numpy as np
from jax import vmap
from matplotlib import pyplot as plt
from lifelike.utils import dump


class CallBack:

    def _compute_loss(self, model, batch, loss):
        X, T, E = batch
        weights = model.get_weights(model.opt_state)
        return loss(weights, (X, T, E))


class Logger(CallBack):
    def __init__(self, report_every_n_epochs=10):
        self.report_every_n_epochs = report_every_n_epochs

    def __call__(self, epoch, model, training_batch=None, loss=None, **kwargs):
        if epoch % self.report_every_n_epochs == 0:
            print("Epoch {:d}: Training set accuracy {:f}".format(epoch, self._compute_loss(model, training_batch, loss)))


class PlotSurvivalCurve(CallBack):
    def __init__(self, individual, update_every_n_epochs=250):
        self.individual = individual
        self.update_every_n_epochs = update_every_n_epochs
        plt.ion()

    def __call__(self, epoch, model, training_batch=None, **kwargs):
        if epoch % self.update_every_n_epochs == 0:
            times = np.linspace(1, 3200, 3200)
            X, _, _ = training_batch
            y = model.predict_survival_function(X[self.individual], times)
            plt.plot(times, y, c="k", alpha=0.15)
            plt.axvline(
                T[self.individual], 0, 1, ls="-" if E[self.individual] else "--"
            )
            plt.draw()
            plt.pause(0.0001)


class ModelCheckpoint(CallBack):
    def __init__(self, filepath, save_every_n_epochs=50, prefix_timestamp=True):
        self.filepath = filepath
        self.save_every_n_epochs = save_every_n_epochs
        self.prefix_timestamp = prefix_timestamp

    def __call__(self, epoch, model, **kwargs):
        if epoch % self.save_every_n_epochs == 0:

            filepath = (
                self._prepend_timestamp(self.filepath)
                if self.prefix_timestamp
                else self.filepath
            )

            dump(model, filepath)
            print("Saved model to %s." % filepath)

    @staticmethod
    def _prepend_timestamp(filepath):
        path = Path(filepath)
        return path.with_name("%d_" % int(time.time()) + path.name)


class EarlyStopping(CallBack):
    """
    This can be improved with:
    1. smoothing average of last few trials
    2. a burn in period (i.e. ignore first 1000x epochs)

    """

    def __init__(self, rdelta=0.25):
        self.rdelta = rdelta
        self.best_test_loss = np.inf
        self.best_train_loss = np.inf

    def __call__(self, epoch, model, training_batch=None, loss=None, **kwargs):
        train_acc = self._compute_loss(model, training_batch, loss)

        if train_acc < self.best_train_loss:
            self.best_train_loss = train_acc
        else:
            if (train_acc - self.best_train_loss) / self.best_train_loss > self.rdelta:
                print("Stopping early as metric is diverging.")
                raise StopIteration()


class TerminateOnNaN(CallBack):


    def __call__(self, epoch, model, training_batch=None, loss=None, **kwargs):
        if np.isnan(self._compute_loss(model, training_batch, loss)):
            print("Stopping early due to NaNs.")
            raise StopIteration()
