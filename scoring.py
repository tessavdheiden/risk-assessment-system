from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

root = os.getcwd()

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])


# Function for model scoring
def score_model():
    # this function should take a trained model, load test data, and calculate
    # an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file
    modelpath = os.path.join(root, output_model_path, 'trainedmodel.pkl')
    with open(modelpath, 'rb') as file:
        model = pickle.load(file)

    folderpath = os.path.join(root, test_data_path)
    pathfile = folderpath + '/' + \
        [name for name in os.listdir(folderpath) if '.csv' in name][0]
    X = pd.read_csv(pathfile)
    X.drop('corporation', inplace=True, axis=1)
    y = X.pop('exited')

    predicted = model.predict(X)
    f1score = metrics.f1_score(predicted, y)
    return f1score


if __name__ == '__main__':
    f1score = score_model()
    filepath = root + '/' + output_model_path + '/' + 'latestscore.txt'

    with open(filepath, 'w+') as f:
        f.write(str(f1score) + '\n')



