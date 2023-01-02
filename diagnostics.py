import ast
import pandas as pd
import pickle
import numpy as np
import timeit
import os
import json
import subprocess
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

root = os.getcwd()
dataset_csv_path = os.path.join(root, config['output_folder_path'])
test_data_path = os.path.join(root, config['test_data_path'])
prod_deployment_path = os.path.join(root, config['prod_deployment_path'])


# Function to get model predictions
def model_predictions(X: pd.DataFrame):
    # read the deployed model and a test dataset, calculate predictions
    # return value should be a list containing all predictions
    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)

    predicted = list(model.predict(X))
    assert len(predicted) == len(X)
    return predicted


# Function to get summary statistics
def dataframe_summary():
    # calculate summary statistics here
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    numeric_cols = [
        'lastmonth_activity',
        'lastyear_activity',
        'number_of_employees']
    df = df[numeric_cols]
    result_summary = df.agg(['mean', 'median', 'std'])
    logger.info(f'Result summary created: {result_summary}')
    assert result_summary.shape == (3, len(numeric_cols))
    return result_summary.values.tolist()


def missing_data():
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    missing_values_df = df.isna().sum() / df.shape[0]
    assert len(missing_values_df) == len(df.columns)
    return missing_values_df


def ingestion_timing(n_times=int(1e0)):
    code = "os.system('python3 ingestion.py')"
    timing = timeit.timeit(code, setup='import os', number=n_times) / n_times
    logger.info(f'Ingestion time: {timing}')
    return timing


def training_timing(n_times=int(1e0)):
    code = "os.system('python3 training.py')"
    timing = timeit.timeit(code, setup='import os', number=n_times) / n_times
    logger.info(f'Training time: {timing}')
    return timing


# Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py
    training_time = training_timing()
    ingestion_time = ingestion_timing()

    # return a list of 2 timing values in seconds
    return [training_time, ingestion_time]


# Function to check dependencies
def outdated_packages_list():
    # get a list of current versions
    with open(os.path.join(root, 'requirements.txt'), 'r') as f:
        requirements = f.read()
    packages = [name.split("=")[0] for name in requirements.split('\n')]
    current_versions = [name.split("=")[-1]
                        for name in requirements.split('\n')]

    # get list of outdated versions
    outdated = subprocess.check_output(['pip', 'list', '--outdated'])
    # convert bytes to string
    outdated_packages = outdated.decode('ascii').split("\n")
    # remove "wheel" and multiple spaces
    outdated_packages = [",".join(package.replace(
        'wheel', "").split()) for package in outdated_packages][2:-1]
    latest_versions = [package.split(",")[-1] for package in outdated_packages]
    outdated_packages = [package.split(",")[0]
                         for package in outdated_packages]

    # only appending outdated versions, so not versions that are up-to-date
    results = []
    for package, current_version in zip(packages, current_versions):
        for outdated_package, latest_version in zip(
                outdated_packages, latest_versions):
            if package == outdated_package:
                results.append([package, current_version, latest_version])

    return results


if __name__ == '__main__':
    X = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X.drop(['corporation', 'exited'], inplace=True, axis=1)

    model_predictions(X)
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()
