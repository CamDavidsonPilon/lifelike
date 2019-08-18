
import numpy as np
from jax import vmap
from matplotlib import pyplot as plt


class CallBack:
    pass



class Logger(CallBack):

    def __init__(self, report_every_n_epochs=1):
        self.report_every_n_epochs = report_every_n_epochs

    def __call__(self, epoch, model, batch=None, loss=None, **kwargs):
        if epoch % self.report_every_n_epochs == 0:
            X, T, E = batch
            weights = model.get_weights(model.opt_state)
            train_acc = loss(weights, (X, T, E))
            print("Epoch {:d}: Training set accuracy {:f}".format(epoch, train_acc))




class PlotSurvivalCurve(CallBack):

    def __init__(self, individual, update_every_n_epochs=250):
        self.individual = individual
        self.update_every_n_epochs = update_every_n_epochs
        plt.ion()

    def __call__(self, epoch, model, batch=None, **kwargs):
        if epoch % self.update_every_n_epochs == 0:
            times = np.linspace(1, 3200, 3200)
            X, T, E = batch
            y = model.predict_survival_function(X[self.individual], times)
            plt.plot(times, y, c='k', alpha=0.15)
            plt.axvline(T[self.individual], 0, 1, ls="-" if E[self.individual] else "--")
            plt.draw()
            plt.pause(0.0001)

