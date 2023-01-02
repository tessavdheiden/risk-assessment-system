from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import utils
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

root = os.getcwd()

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(root, config['output_model_path'])


# Function for model scoring
def score_model():
    # this function should take a trained model, load test data, and calculate
    # an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file
    modelpath = os.path.join(output_model_path, 'trainedmodel.pkl')
    model = utils.load_model(modelpath)

    X = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    y = X["exited"]
    dropped_columns = ["exited", "corporation"]
    X = X.drop(dropped_columns, axis=1)

    predicted = model.predict(X)
    f1score = metrics.f1_score(predicted, y)

    logger.info(f'Model score: {f1score:.2f}')
    return f1score


if __name__ == '__main__':
    f1score = score_model()
    filepath = os.path.join(output_model_path, 'latestscore.txt')

    logger.info(f'Model score saved to: {filepath}')
    with open(filepath, 'w+') as f:
        f.write(str(f1score) + '\n')



