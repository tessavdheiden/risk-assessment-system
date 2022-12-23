import ast
import pandas as pd
import pickle
import numpy as np
import timeit
import os
import json
import subprocess

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

root = os.getcwd()
dataset_csv_path = os.path.join(root, config['output_folder_path'])
test_data_path = os.path.join(root, config['test_data_path'])
prod_deployment_path = os.path.join(root, config['prod_deployment_path'])

# Function to get model predictions


def model_predictions():
    # read the deployed model and a test dataset, calculate predictions
    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)

    X = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X.drop(['corporation', 'exited'], inplace=True, axis=1)

    # return value should be a list containing all predictions
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
    means = df.mean()
    medians = df.median()
    stds = df.std()
    final_list = np.concatenate([means.values, medians.values, stds.values])
    return final_list


def missing_data():
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    nans = df.isna().sum()
    percent = [nan / len(df) for nan in nans]
    assert len(percent) == len(df.columns)
    return percent


def ingestion_timing():
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing = timeit.default_timer() - starttime
    return timing


def training_timing():
    starttime = timeit.default_timer()
    os.system('python3 training.py')
    timing = timeit.default_timer() - starttime
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

    return result


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()
