import logging
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt


class CategoricalScores:

    @staticmethod
    def get_TPTN(obs, pred):
        if len(obs) == len(pred):
            if all(O in [0, 1] for O in obs) and all(P in [0, 1] for P in pred):
                output_list = []
                test = {
                    (0, 0): "TN",
                    (1, 1): "TP",
                    (0, 1): "FN",
                    (1, 0): "FP"
                }
                for O, P in zip(obs, pred):
                    output_list.append(test.get((O, P)))
                return output_list
            else:
                print('Invalid values present in Y-Observed and/or Y-Predicted.'
                      'Only binary values 0 and 1 are supported!')
        else:
            print('Invalid list size. Y-Observed and Y-Predicted must match!')

    def metrics(self, obs, pred):

        tptn = self.get_TPTN(obs, pred)
        tn = tptn.count('TN')
        tp = tptn.count('TP')
        fn = tptn.count('FN')
        fp = tptn.count('FP')
        accuracy = (tp + tn) / len(tptn)
        bias = (tp + fp) / (tp + fn)
        pod = tp / (tp + fn)
        pofd = fp / (fp + tn)
        far = fp / (tp + fp)
        csi = tp / (tp + fp + fn)
        ph = ((tp + tn) * (tp + fp)) / len(tptn)
        ets = (tp - ph) / (tp + fp + fn - ph)
        hss = ((tp * tn) - (fp * fn)) / ((((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))) / 2)
        hkd = pod - pofd
        num_pixels = len(obs)
        return accuracy, bias, pod, pofd, far, csi, ph, ets, hss, hkd, num_pixels


class ContinuousScores:
    @staticmethod
    def metrics(obs, pred):
        y_pred_mean = np.nanmean(pred)
        y_true_mean = np.nanmean(obs)
        mae = mean_absolute_error(obs, pred)
        rmse = sqrt(mean_squared_error(obs, pred))
        std = sqrt(np.nanmean((pred- mae)**2))
        fseperc = (rmse/y_true_mean)*100
        fse = rmse/y_true_mean
        corr = np.corrcoef(obs, pred)
        num_pixels = len(obs)
        return y_pred_mean, y_true_mean, mae, rmse, std, fseperc, fse, corr, num_pixels

