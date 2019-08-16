


class CallBack:
    pass



class Logger(CallBack):

    def __init__(self, report_every_n_epochs=1):
        self.report_every_n_epochs = report_every_n_epochs

    def __call__(self, epoch, opt_state, get_weights, batch, loss):
        if epoch % self.report_every_n_epochs == 0:
            X, T, E = batch
            weights = get_weights(opt_state)
            train_acc = loss(weights, (X, T, E))
            print("Epoch {:d}: Training set accuracy {:f}".format(epoch, train_acc))
