from __future__ import absolute_import, division, print_function

import os
import sys

import logging
import pandas as pd
import numpy as np

from pprint import pformat

from core import utils
from core import validation
from core import graphics as g

from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tensorflow import keras

from keras.layers import GaussianNoise
from keras.models import Sequential
from keras.layers import Dense


class Training:
    """
    This module is intended to automate the TensorFlow Neural Network training.

    """
    INPUT_DATA = pd.DataFrame()
    NN_RUN = 0
    OUTPUT_DIR = 'output/'
    RANDOM_SEED = 0

    def __init__(self, INPUT_DATA=None, PCTAG='notag', NN_RUN=0, OUTPUT_DIR='output/', RANDOM_SEED=0):

        self.INPUT_DATA = INPUT_DATA
        self.PCTAG = PCTAG
        self.NN_RUN = NN_RUN
        self.OUTPUT_DIR = OUTPUT_DIR
        self.RANDOM_SEED = RANDOM_SEED
        self.grph = g.Graphics(self.PCTAG, self.NN_RUN, self.OUTPUT_DIR)

    # --------------------------------------------------------------------------
    # BUILD NN MODELS - DEFINITIONS : CLAS = CLASSIFICATION and REG = REGRESSION
    # --------------------------------------------------------------------------
    @staticmethod
    def build_class_model():
        """
        Fucntion to create the instance and configuration of the keras
        model(Sequential and Dense).
        """
        # Create the Keras model:
        model = Sequential()
        model.add(Dense(8, input_dim=4, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(2, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    @staticmethod
    def build_reg_model(input_size):
        """
        Fucntion to create the instance and configuration of the keras
        model(Sequential and Dense).
        """
        model = Sequential()
        model.add(GaussianNoise(0.01, input_shape=(input_size,)))
        model.add(Dense(33, activation='linear'))
        model.add(Dense(11, activation='linear'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mean_absolute_error', 'mean_squared_error'])
        return model

    def train_screening_net(self):
        """

        :return:
        """
        # Fix random seed for reproducibility:
        np.random.seed(self.RANDOM_SEED)

        df = self.INPUT_DATA

        # SUBSET BY SPECIFIC CLASS (UNDERSAMPLING)
        n = 0.98

        to_remove = np.random.choice(
            df.index,
            size=int(df.shape[0] * n),
            replace=False)

        df = df.drop(to_remove)

        # Test for valid input data
        expected_columns = ['36V', '89V', '166V', '190V', 'TagRain']

        if not set(expected_columns).issubset(set(list(df.columns.values))):
            logging.info(f'Some of the expected columns where not present in the input dataframe.')
            logging.info(f'\nExpected columns:'
                         f'\n{expected_columns}\n'
                         f'Found columns:'
                         f'\n{list(df.columns.values)}\n'
                         f'System halt by unmet conditions.')
            sys.exit(1)

        x, y = df.loc[:, ['36V', '89V', '166V', '190V']], df.loc[:, ['TagRain']]

        x_arr = np.asanyarray(x)
        y_arr = np.asanyarray(y)
        y_arr = np.ravel(y_arr)

        # Scaling the input paramaters:
        norm_sc = Normalizer()
        x_normalized = norm_sc.fit_transform(x_arr)

        # Split the dataset in test and train samples:
        x_train, x_test, y_train, y_test = train_test_split(x_normalized,
                                                            y_arr, test_size=0.10,
                                                            random_state=self.RANDOM_SEED)

        # Create the instance for KerasRegressor:
        model = self.build_class_model()

        # ------------------------------------------------------------------------------
        # Display training progress by printing a single dot for each completed epoch

        class PrintDot(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                if epoch % 10 == 0: print('')
                print('.', end='')

        EPOCHS = 100

        history = model.fit(x_train, y_train,
                            epochs=EPOCHS, validation_split=0.2, batch_size=10,
                            verbose=0, callbacks=[PrintDot()])

        # ------------------------------------------------------------------------------
        # Visualize the model's training progress using the stats
        # stored in the history object.
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        hist.tail()

        # ------------------------------------------------------------------------------
        # Saving the complete model in HDF5
        # model.save(self.OUTPUT_DIR + '' + str(self.NN_RUN) + '.h5')
        return model

    def train_retrieval_net(self):

        # Fix random seed for reproducibility
        np.random.seed(self.RANDOM_SEED)
        # ------------------------------------------------------------------------------

        # Load dataset:
        df = self.INPUT_DATA
        # Test for valid input data
        expected_columns = ['10V', '10H', '18V', '18H', '36V', '36H', '89V', '89H',
                            '166V', '166H', '183VH', 'sfccode', 'T2m', 'tcwv', 'PCT36',
                            'PCT89', '89VH', 'lat', 'sfcprcp']

        if not set(expected_columns).issubset(set(list(df.columns.values))):
            logging.info(f'Some of the expected columns where not present in the input dataframe.')
            logging.info(f'\nExpected columns:'
                         f'\n{expected_columns}\n'
                         f'Found columns:'
                         f'\n{list(df.columns.values)}\n'
                         f'System halt by unmet conditions.')
            sys.exit(1)
        # ------------------------------------------------------------------------------
        # Extracting attributes of interest from the input dataframe
        df_input = df.loc[:, ['10V', '10H', '18V', '18H', '36V', '36H', '89V', '89H',
                              '166V', '166H', '183VH', 'sfccode', 'T2m', 'tcwv', 'PCT36',
                              'PCT89', '89VH', 'lat']]

        # Saving attribute column names for future use
        colunas = list(df_input.columns.values)

        # Creating instance of the data standardization class
        scaler = StandardScaler()

        # Standardising input attributes
        input_x_norm = scaler.fit_transform(df_input)
        
        # Converting from vector to pandas dataframe
        df_normed_input = pd.DataFrame(input_x_norm[:],columns=colunas)
        
        # Splitting the ancillary dataframe  
        ancillary = df_normed_input.loc[:, ['183VH', 'sfccode', 'T2m', 'tcwv', 'PCT36', 'PCT89', '89VH', 'lat']]

        # Splitting attributes to further apply the PCA
        tb_low = df_normed_input.loc[:, ['10V', '10H', '18V', '18H']]
        tb_high = df_normed_input.loc[:, ['36V', '36H', '89V', '89H', '166V', '166H']]

        # Calculating the attributes contribution to the PCA
        pca = PCA()
        pca1 = pca.fit(tb_low)

        # Generating and saving figures
        self.grph.plot_pca_contrib(pca1, 'low')

        # Applying PCA transform on the low frequency channels
        pca_trans1 = PCA(n_components=2)
        pca_trans1.fit(tb_low)
        tb_low_transformed = pca_trans1.transform(tb_low)
        logging.info("PCA original shape:   ", tb_low.shape)
        logging.info("PCA transformed shape:", tb_low_transformed.shape)

        # Repeating the above steps for the second PCA
        pca = PCA()
        pca2 = pca.fit(tb_high)

        # Generating and saving figures
        self.grph.plot_pca_contrib(pca2, 'high')

        # Applying PCA transform on the low frequency channels
        pca_trans2 = PCA(n_components=2)
        pca_trans2.fit(tb_high)
        tb_high_transformed = pca_trans2.transform(tb_high)
        logging.info("original shape:   ", tb_high.shape)
        logging.info("transformed shape:", tb_high_transformed.shape)

        # Join both PCAs back into one sigle dataset
        PCA1 = pd.DataFrame(tb_low_transformed[:], columns=['pca1_1', 'pca_2'])
        PCA2 = pd.DataFrame(tb_high_transformed[:], columns=['pca2_1', 'pca2_2'])

        dataset = PCA1.join(PCA2, how='right')
        dataset = dataset.join(ancillary, how='right')
        dataset = dataset.join(df.loc[:, ['sfcprcp']], how='right')

        # SUBSET BY PRECIPITATION RANGE
        dataset = utils.keep_df_interval(0.2, 60, dataset, 'sfcprcp')

        # Split the data into train and test
        # Now split the dataset into a training set and a test set.
        # We will use the test set in the final evaluation of our model.
        train_dataset = dataset.sample(frac=0.8, random_state=self.RANDOM_SEED)
        test_dataset = dataset.drop(train_dataset.index)

        # Also look at the overall statistics:
        train_stats = train_dataset.describe()
        train_stats.pop("sfcprcp")
        train_stats = train_stats.transpose()

        # Split features from labels:
        # Separate the target value, or "label", from the features.
        # This label is the value that you will train the model to predict.
        y_train = train_dataset.pop('sfcprcp')
        y_test = test_dataset.pop('sfcprcp')

        # Normalize the data:
        scaler = StandardScaler()
        normed_train_data = scaler.fit_transform(train_dataset)
        normed_test_data = scaler.fit_transform(test_dataset)

        # Build the model:
        model = self.build_reg_model(len(train_dataset.keys()))

        # Inspect the model:
        # Use the .summary method to print a simple description of the model
        model.summary()

        # Display training progress by printing a single dot for each completed epoch
        class PrintDot(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                if epoch % 10 == 0: print('')
                print('.', end='')

        EPOCHS = 100

        history = model.fit(
            normed_train_data, y_train,
            epochs=EPOCHS, validation_split=0.2, verbose=0,
            callbacks=[PrintDot()])
        print(history.history.keys())

        # Visualize the model's training progress using the stats
        # stored in the history object.
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        hist.tail()

        self.grph.plot_history(history)

        model = self.build_reg_model(len(train_dataset.keys()))

        # The patience parameter is the amount of epochs to check for improvement
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

        history = model.fit(normed_train_data, y_train, epochs=EPOCHS,
                            validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

        # Ploting again, but with the EarlyStopping apllied:
        self.grph.plot_history_early_stopping(history)

        # Let's see how well the model generalizes by using
        # the test set, which we did not use when training the model.
        # This tells us how well we can expect the model to predict
        # when we use it in the real world.

        loss, mae, mse = model.evaluate(normed_test_data, y_test, verbose=0)

        logging.info("Testing set Mean Abs Error: {:5.2f} sfcprcp".format(mae))

        # Make predictions
        # Finally, predict SFCPRCP values using data in the testing set
        test_predictions = model.predict(normed_test_data).flatten()

        # Appplying meteorological skills to verify the performance
        # of the TRAIN/TESTE model, in this case, continous scores
        skills = validation.ContinuousScores()
        val_y_pred_mean, val_y_test_mean, val_mae, val_rmse, val_std, val_fseperc, val_fse, val_corr, val_num_pixels = skills.metrics(
            y_test, test_predictions)

        # Converting validation statistics to dictionary
        my_scores = {'val_y_pred_mean': val_y_pred_mean,
                     'val_y_test_mean': val_y_test_mean,
                     'val_mae': val_mae,
                     'val_rmse': val_rmse,
                     'val_std': val_std,
                     'val_fseperc': val_fseperc,
                     'val_fse': val_fse,
                     'val_corr': val_corr,
                     'val_num_pixels': val_num_pixels}

        my_scores = pformat(my_scores)

        logging.info(f'Validation statistics:\n\n{my_scores}\n')

        # Saving scores dictionary
        with open(f'{self.OUTPUT_DIR}{self.PCTAG}v{self.NN_RUN}_continuous_scores_TEST_TRAIN.txt', 'w') as myfile:
            myfile.write(my_scores)

        logging.info(f'Retrieval validation statistics file saved at:\n\n'
                     f'{self.OUTPUT_DIR}{self.PCTAG}v{self.NN_RUN}_continuous_scores_TEST_TRAIN.txt')

        # Plot & save : scatter test vs pred
        self.grph.plot_scatter_test_vs_pred(y_test, test_predictions)

        # Plot & save : LOG scatter test vs pred
        self.grph.plot_scatter_log_test_vs_pred(y_test, test_predictions)

        # Plot & save : error distribution
        error = test_predictions - y_test
        self.grph.plot_prediction_error(error)

        # Plot & save : hist2d
        self.grph.plot_hist2d(y_test, test_predictions)

        return model

