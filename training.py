from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

root = os.getcwd()
dataset_csv_path = os.path.join(root, config['output_folder_path'])
model_path = os.path.join(root, config['output_model_path'])


# Function for training the model
def train_model(model_save_path):

    # use this logistic regression for training
    lr = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)

    # fit the logistic regression to your data
    X = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    y = X["exited"]
    dropped_columns = ["exited", "corporation"]
    X = X.drop(dropped_columns, axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, y)
    logger.info('Training model...')
    lr.fit(X_train, y_train)

    # write the trained model to your workspace in a file called
    # trainedmodel.pkl
    logger.info(f'Saving trained model {"/".join(model_save_path.split("/")[-2:])}')
    filehandler = open(model_save_path, 'wb')
    pickle.dump(lr, filehandler)


if __name__ == '__main__':
    outputpath = root + '/' + model_path
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    model_name = 'trainedmodel.pkl'
    train_model(model_save_path=os.path.join(model_path, model_name))