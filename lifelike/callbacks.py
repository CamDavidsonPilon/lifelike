
import numpy as np
from matplotlib import pyplot as plt


class CallBack:
    pass



class Logger(CallBack):

    def __init__(self, report_every_n_epochs=1):
        self.report_every_n_epochs = report_every_n_epochs

    def __call__(self, epoch, opt_state=None, get_weights=None, batch=None, loss_function=None, **kwargs):
        if epoch % self.report_every_n_epochs == 0:
            X, T, E = batch
            weights = get_weights(opt_state)
            train_acc = loss_function(weights, (X, T, E))
            print("Epoch {:d}: Training set accuracy {:f}".format(epoch, train_acc))




class PlotSurvivalCurve(CallBack):

    def __init__(self, individual, update_every_n_epochs=250):
        self.individual = individual
        self.update_every_n_epochs = update_every_n_epochs
        plt.ion()

    def __call__(self, epoch, opt_state=None, get_weights=None, batch=None, loss=None, predict=None, **kwargs):
        if epoch % self.update_every_n_epochs == 0:
            times = np.linspace(1, 2000, 2000)
            X, T, E = batch
            weights = get_weights(opt_state)
            y = loss.survival_function(predict(weights, X[[0]]), times)
            plt.plot(times, y, c='k', alpha=0.15)
            plt.draw()
            plt.pause(0.0001)

