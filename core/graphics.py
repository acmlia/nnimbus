import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------
# FUNCTIONS TO MAKE PLOTS ABOUT TRAINING:
# -------------------------------------------------------------------------


class Graphics:

    FILENAMETAG = 'output/'

    def __init__(self, FILENAMETAG='notag'):
        self.FILENAMETAG = FILENAMETAG

    def _makename(self, fig_name):
        name = f'{self.FILENAMETAG}_{fig_name}.png'
        return name

    def plot_pca_contrib(self, pca, lowhi):
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of components for tb_low')
        plt.ylabel('Cumulative explained variance')
        plt.savefig(self._makename('pca_tb' + lowhi))

    def plot_history(self, history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [sfcprcp]')
        plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val Error')
        ylim_max = hist.val_mean_absolute_error.max() + 10
        plt.ylim([0, ylim_max])
        plt.legend()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$scfprcp^2$]')
        plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
        ylim_max = hist.val_mean_squared_error.max() + 10
        plt.ylim([0, ylim_max])
        plt.legend()
        plt.savefig(self._makename('error_per_epochs_history'))

    def plot_history_early_stopping(self, history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [sfcprcp]')
        plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val Error')
        ylim_max = hist.val_mean_absolute_error.max() + 10
        plt.ylim([0, ylim_max])
        plt.legend()
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$sfcprcp^2$]')
        plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
        ylim_max = hist.val_mean_squared_error.max() + 10
        plt.ylim([0, ylim_max])
        plt.legend()
        plt.savefig(self._makename('error_per_epochs_earlystopping'))

    def plot_hist2d(self, y_test, test_predictions):
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
        plt.savefig(self._makename('hist2D'))
        plt.clf()

    def plot_scatter_test_vs_pred(self, y_test, test_predictions):
        plt.figure()
        plt.scatter(y_test, test_predictions)
        plt.xlabel('True Values [sfcprcp]')
        plt.ylabel('Predictions [sfcprcp]')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0, plt.xlim()[1]])
        plt.ylim([0, plt.ylim()[1]])
        plt.plot([-100, 100], [-100, 100])
        plt.savefig(self._makename('scatter_y_test_vs_y_pred'))
        plt.clf()

    def plot_scatter_log_test_vs_pred(self, y_test, test_predictions):
        ax = plt.gca()
        ax.plot(y_test, test_predictions, 'o', c='blue', alpha=0.07, markeredgecolor='none')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('True Values [sfcprcp]')
        ax.set_ylabel('Predictions [sfcprcp]')
        plt.plot([-100, 100], [-100, 100])
        plt.savefig(self._makename('LOG_scatter_y_test_vs_y_pred'))
        plt.clf()

    def plot_prediction_error(self, error):
        plt.hist(error, bins=25)
        plt.xlabel("Prediction Error [sfcprcp]")
        plt.ylabel("Count")
        plt.savefig(self._makename('prediction_error'))
        plt.clf()
