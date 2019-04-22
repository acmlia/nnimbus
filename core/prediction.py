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


class Prediction:
    """
    This module is intended to automate the TensorFlow Neural Network training.

    """
    def __init__(self, INPUT_DATA=None, RANDOM_SEED=0, FILENAMETAG='output/'):

        self.INPUT_DATA = INPUT_DATA
        self.RANDOM_SEED = RANDOM_SEED
        self.FILENAMETAG = FILENAMETAG
        self.grph = g.Graphics(FILENAMETAG)

    def status(self):
        """
        Shows the settings of the main parameters necessary to process the algorithm.
        """
        logging.info(f'{__name__} OK\n'
                     f'INPUT_DATA = {self.INPUT_DATA}\n'
                     f'RANDOM_SEED = {self.RANDOM_SEED}\n'
                     f'FILENAMETAG = {self.FILENAMETAG}\n')

    def save_stuff(self):
        with open(f'{self.FILENAMETAG}_arquivo_de_teste.txt', 'w') as myfile:
            myfile.write('bla bla bla bla')

    def save_outside(self, test_string):
        test_string = test_string + 'BLA BLA BLA'
        return test_string