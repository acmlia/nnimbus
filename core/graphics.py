import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------
# FUNCTIONS TO MAKE PLOTS ABOUT TRAINING:
# -------------------------------------------------------------------------


def plot_history(self, history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [sfcprcp]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    ylim_max = hist.val_mean_absolute_error.max() + 10
    plt.ylim([0, ylim_max])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$scfprcp^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    ylim_max = hist.val_mean_squared_error.max() + 10
    plt.ylim([0, ylim_max])
    plt.legend()
    fig_name = self.fig_title + "_error_per_epochs_history.png"
    plt.savefig(self.path_fig + fig_name)


def plot_history_early_stopping(self, history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [sfcprcp]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    ylim_max = hist.val_mean_absolute_error.max() + 10
    plt.ylim([0, ylim_max])

    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$sfcprcp^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    ylim_max = hist.val_mean_squared_error.max() + 10
    plt.ylim([0, ylim_max])

    plt.legend()

    fig_name = self.fig_title + "_error_per_epochs_EarlyStopping.png"
    plt.savefig(self.path_fig + fig_name)


def plot_hist2d(y_test, test_predictions):
    plt.hist2d(y_test, test_predictions, cmin=1, bins=(50, 50), cmap=plt.cm.jet,
               range=np.array([(0.2, 110), (0.2, 110)]))
    plt.axis('equal')
    plt.axis('square')
    plt.plot([0, 100], [0, 100], ls="--", c=".3")
    plt.xlim([0, max(y_test)])
    plt.ylim([0, max(y_test)])
    plt.colorbar()
    plt.xlabel("Observed rain rate (mm/h) - Training")
    plt.ylabel("Predicted rain rate (mm/h) - Training")
    figure = plt
    return figure
