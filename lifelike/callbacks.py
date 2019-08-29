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
    """
    Print out to stdout important metrics.

    TODO: include epoch training time


    """
    def __init__(self, report_every_n_epochs=10):
        self.report_every_n_epochs = report_every_n_epochs

    def __call__(self, epoch=None, model=None, training_batch=None, testing_batch=None, loss=None, **kwargs):
        if epoch % self.report_every_n_epochs == 0:
            print("Epoch {:d}: training set accuracy {:f}, testing set accuracy {:f}".format(epoch, self._compute_loss(model, training_batch, loss), self._compute_loss(model, testing_batch, loss)))


class PlotSurvivalCurve(CallBack):
    """
    Display the predicted survival function for individuals. Useful for debugging.


    Warning
    --------

    Doesn't really work with batch size < total size

    """
    def __init__(self, individuals, update_every_n_epochs=250):
        self.individuals = individuals
        self.update_every_n_epochs = update_every_n_epochs
        plt.ion()

    def __call__(self, epoch=None, model=None, training_batch=None, **kwargs):
        if epoch % self.update_every_n_epochs == 0 and epoch > 0:
            times = np.linspace(1, 120, 150)
            X, T, E = training_batch
            colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(self.individuals) + 2)))

            for individual, color in zip(self.individuals, colors):
                y = model.predict_survival_function(X[individual], times)
                plt.plot(times, y, c=color, alpha=0.20)

                if epoch == self.update_every_n_epochs:
                    # only need to plot once
                    plt.axvline(
                        T[individual], 0, 1, ls="-" if E[individual] else "--", c=color, alpha=0.85
                    )
            plt.draw()
            plt.pause(0.0001)


class ModelCheckpoint(CallBack):
    """
    Save the model, including weights and loss, to disk.

    """
    def __init__(self, filepath, save_every_n_epochs=50, prefix_timestamp=True):
        self.filepath = filepath
        self.save_every_n_epochs = save_every_n_epochs
        self.prefix_timestamp = prefix_timestamp

    def __call__(self, epoch=None, model=None, **kwargs):
        if epoch % self.save_every_n_epochs == 0 and epoch > 0:

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

    def __call__(self, epoch=None, model=None, training_batch=None, loss=None, **kwargs):
        train_acc = self._compute_loss(model, training_batch, loss)

        if train_acc < self.best_train_loss:
            self.best_train_loss = train_acc
        else:
            if (train_acc - self.best_train_loss) / self.best_train_loss > self.rdelta:
                print("Stopping early as metric is diverging.")
                raise StopIteration()


class TerminateOnNaN(CallBack):
    """
    If NaNs are detected, the training is stopped.

    """

    def __call__(self, epoch=None, model=None, training_batch=None, loss=None, **kwargs):
        if np.isnan(self._compute_loss(model, training_batch, loss)):
            print("Stopping early due to NaNs.")
            raise StopIteration()
