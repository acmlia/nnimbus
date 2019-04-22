#! /usr/bin/env python3

# ,----------------,
# | PYTHON IMPORTS |----------------------------------------------------------------------------------------------------
# '----------------'

import logging
import os

from pprint import pformat
from decouple import config
from decouple import Csv

from core import utils
from core.pre_process import PreProcess
from core.prediction import Prediction


# ,------------------------,
# | .env IMPORTED SETTINGS |--------------------------------------------------------------------------------------------
# '------------------------'
"""
Environment settings comes from the external .env file through the use of the python-decouple module.
"""
# Tag with the PC name (please avoid using special characteres)
PCTAG = config('PCTAG', default='default')

# Path for the CSV files
IN_CSV_PATH = config('IN_CSV_PATH', default='')

# Name of the CSV file
IN_CSV_NAME = config('IN_CSV_NAME', default='')

# Path for the directory that contains the CSV files that will be used in subset/concatenation steps
IN_CSV_LIST = config('IN_CSV_LIST', default='')

# Directory where the program should save its outputs
OUTPUT_DIR = config('OUTPUT_DIR', default='')

# Final training data saving name
TRNGCSV_TO_SAVE = config('TRNGCSV_TO_SAVE', default='')

# Random seed for reproducible training
RANDOM_SEED = config('RANDOM_SEED', default=0, cast=int)

# Geographical coordinates for regional subset
LAT_LIMIT = config('LAT_LIMIT', default=0, cast=Csv(float))
LON_LIMIT = config('LON_LIMIT', default=0, cast=Csv(float))

# Minimal threshold of rain rate
THRESHOLD_RAIN = config('THRESHOLD_RAIN', default=0, cast=float)

# ,--------------------------,
# | NNIMBUS WORKFLOW OPTIONS |------------------------------------------------------------------------------------------
# '--------------------------'

# NNIMBUS logs
LOGFILE = OUTPUT_DIR+f'nnimbus_{PCTAG}.log'
VERFILE = OUTPUT_DIR+f'nnimbus_{PCTAG}.ver'

# Setting up information logs for every NNIMBUS execution in an external file
logging.basicConfig(filename=LOGFILE, format='%(asctime)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.DEBUG)

# check for existing version file to retrieve the number of the last NNIMBUS execution
if os.path.isfile(VERFILE):
    with open(VERFILE, 'r+') as f:
        str_current_version = f.readline().rstrip().split('#')[1]
        NN_RUN = int(str_current_version) + 1
        f.seek(0)
        f.write('NN_Run #'+str(NN_RUN))
        log_found_message = f'Version file found. Saving logs at {os.path.abspath(LOGFILE)}'
else:
    NN_RUN = 0
    version_warning = f'> No version file found in "{os.path.abspath(VERFILE)}"\n' \
        f'> Tagging this execution as version: #{NN_RUN}'
    print(version_warning)
    logging.info(version_warning)
    with open(VERFILE, 'w+') as f:
        f.seek(0)
        f.write(f'NN_Run #{NN_RUN}\n')

FILENAMETAG = f'{OUTPUT_DIR}{PCTAG}v{NN_RUN}'

workflow = {
    'read_csv': True,
    'read_raw_csv': False,
    'extract_region': False,
    'concatenate_csv_list_to_df': False,
    'compute_additional_variables': False,
    'load_training': True,
    'train_screening': True,   # requires load_training = True!
    'train_retrieval': False,  # requires load_training = True!
    'pre_process_HDF5': False,
    'prediction': False,
    'validation': False,
    'save_data': False
}

# ,-----------,
# | RUN MODEL |---------------------------------------------------------------------------------------------------------
# '-----------'


def main():
    greatings_header = f'| Starting NNIMBUS @ {PCTAG} # NN_RUN:{NN_RUN} |'
    separator = utils.repeat_to_length('-', len(greatings_header) - 2)
    logging.info(f',{separator},')
    logging.info(greatings_header)
    logging.info(f'\'{separator}\'')
    print(f',{separator},\n' +
          greatings_header + '\n' +
          f'\'{separator}\'')

    # Initializing core classes
    preprocess = PreProcess()
    prediction = Prediction()

    # ,----------------------------,
    # | SAVING MODEL CONFIG TO LOG |------------------------------------------------------------------------------------
    # '----------------------------'
    logging.info(f'User-defined workflow:\n\n{pformat(workflow)}\n')
    logging.info(f'Environment variables:\n\n'
                 f'PCTAG = {PCTAG}\n'
                 f'IN_CSV_PATH = {IN_CSV_PATH}\n'
                 f'IN_CSV_NAME = {IN_CSV_NAME}\n'
                 f'OUTPUT_DIR = {OUTPUT_DIR}\n'
                 f'TRNGCSV_TO_SAVE = {TRNGCSV_TO_SAVE}\n'
                 f'LAT_LIMIT = {LAT_LIMIT}\n'
                 f'LON_LIMIT = {LON_LIMIT}\n'
                 f'THRESHOLD_RAIN = {THRESHOLD_RAIN}\n'
                 f'VERSION_FILE = {os.path.abspath(VERFILE)}\n'
                 f'LOG_FILE = {os.path.abspath(LOGFILE)}\n'
                 f'RANDOM_SEED = {RANDOM_SEED}\n')

    # ,------------------,
    # | Reading CSV Data |----------------------------------------------------------------------------------------------
    # '------------------'
    if workflow['read_csv']:
        logging.info(f'Reading input CSV data')
        training_data = preprocess.load_csv(IN_CSV_PATH, IN_CSV_NAME)

    # ,----------------------,
    # | Reading raw CSV Data |------------------------------------------------------------------------------------------
    # '----------------------'
    if workflow['read_raw_csv']:
        logging.info(f'Reading RAW Randel CSV data')
        training_data = preprocess.load_raw_csv(IN_CSV_PATH, IN_CSV_NAME)

    # ,------------------------------------------,
    # | Extracting region of interest by LAT LON |----------------------------------------------------------------------
    # '------------------------------------------'
    if workflow['extract_region']:
        logging.info(f'Extracting region of interest')
        training_data = preprocess.extract_region(dataframe=training_data,
                                                   lat_min=LAT_LIMIT[0],
                                                   lat_max=LAT_LIMIT[1],
                                                   lon_min=LAT_LIMIT[0],
                                                   lon_max=LAT_LIMIT[1])

    # ,------------------------------,
    # | Concatenating CSV dataframes |----------------------------------------------------------------------------------
    # '------------------------------'
    if workflow['concatenate_csv_list_to_df']:
        logging.info(f'Reading CSV to generate a list of dataframes.')
        training_data = preprocess.load_csv_list(IN_CSV_PATH)
        logging.info(f'Concatenating list of CSV into a single dataframe')
        training_data = preprocess.concatenate_df_list(training_data)

    # ,------------------------------,
    # | Compute additional variables |----------------------------------------------------------------------------------
    # '------------------------------'
    if workflow['compute_additional_variables']:
        logging.info(f'Computing additional variables')
        logging.info(f'Input dataset columns: {list(training_data.columns.values)}')
        training_data = preprocess.compute_additional_input_vars(training_data)
        logging.info(f'Output dataset columns: {list(training_data.columns.values)}')

    # ,-------------------------,
    # | Load training libraries |---------------------------------------------------------------------------------------
    # '-------------------------'
    if workflow['load_training']:
        training_import_warning = f'> Importing training modules, this may take a while...'
        print(training_import_warning)
        logging.info(training_import_warning)

        from core.training import Training

        training = Training(INPUT_DATA=training_data,
                            PCTAG=PCTAG,
                            NN_RUN=NN_RUN,
                            OUTPUT_DIR=OUTPUT_DIR,
                            RANDOM_SEED=RANDOM_SEED)

        training_done_warning = f'> Training modules loaded.'
        print(training_done_warning)
        logging.info(training_done_warning)
    # ,---------------------,
    # | Training: Screening |-------------------------------------------------------------------------------------------
    # '---------------------'
    if workflow['train_screening']:
        screening_warning = f'> Training screening model...'
        print(screening_warning)
        logging.info(screening_warning)

        screening_model = training.train_screening_net()
        # Saving the complete model in HDF5:
        screening_model_name = f'{FILENAMETAG}_screeningmodel.h5'
        screening_model.save(screening_model_name)

        screening_done = f'> Screening model training completed.\n ' \
            f'Output model saved as: {screening_model_name}'
        print(screening_done)
        logging.info(screening_done)
    # ,---------------------,
    # | Training: Retrieval |-------------------------------------------------------------------------------------------
    # '---------------------'
    if workflow['train_retrieval']:
        retrieval_warning = f'> Training retrieval model...'
        print(retrieval_warning)
        logging.info(retrieval_warning)

        retrieval_model = training.train_retrieval_net()

        # Saving model to YAML:
        model_yaml = retrieval_model.to_yaml()
        with open(f'{FILENAMETAG}.yaml', 'w') as yaml_file:
            yaml_file.write(model_yaml)

        # serialize weights to HDF5
        retrieval_model.save_weights(f'{FILENAMETAG}_retrieval_weights.h5')

        # Saving the complete model in HDF5:
        retrieval_model.save(f'{FILENAMETAG}_retrieval_model.h5')

        retrieval_done = f'> Screening model training completed.\n ' \
            f'Output files saved at: {os.path.abspath(OUTPUT_DIR)}/'
        print(retrieval_done)
        logging.info(retrieval_done)

    # ,-----------,
    # | Read HDF5 |-----------------------------------------------------------------------------------------------------
    # '-----------'
    if workflow['pre_process_HDF5']:
        logging.info(f'Reading HDF5')

    # ,------------,
    # | Prediction |----------------------------------------------------------------------------------------------------
    # '------------'
    if workflow['prediction']:
        logging.info(f'Predicting stuff')

    # ,------------,
    # | Validation |----------------------------------------------------------------------------------------------------
    # '------------'
    if workflow['validation']:
        logging.info(f'Validating stuff')

    # ,-----------,
    # | Save data |----------------------------------------------------------------------------------------------------
    # '-----------'
    if workflow['save_data']:
        logging.info(f'Saving stuff')
        file_name = TRNGCSV_TO_SAVE
        utils.tic()
        training_data.to_csv(os.path.join(OUTPUT_DIR, file_name), index=False, sep=",", decimal='.')
        t_hour, t_min, t_sec = utils.tac()
        logging.info(f'Dataframe successfully saved as CSV in {t_hour}h:{t_min}m:{t_sec}s')


if __name__ == '__main__':
    utils.tic()
    main()
    t_hour, t_min, t_sec = utils.tac()
    final_message = f'| Elapsed execution time: {t_hour}h : {t_min}m : {t_sec}s |'
    separator = utils.repeat_to_length('-', len(final_message) - 2)
    logging.info(f',{separator},')
    logging.info(final_message)
    logging.info(f'\'{separator}\'')
