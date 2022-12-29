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

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


# Function for training the model
def train_model():

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
    root = os.getcwd()
    folderpath = root + '/' + dataset_csv_path
    pathfile = folderpath + '/' + \
        list(filter(lambda x: '.csv' in x, os.listdir(folderpath)))[0]
    X = pd.read_csv(pathfile)
    X = X.drop('corporation', axis=1)
    # X = df[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    # y = df['exited'].values.reshape(-1, 1)
    # this removes the column "exited" from X and puts it into y
    y = X.pop("exited")
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    logger.info('Training model...')
    lr.fit(X_train, y_train)

    # write the trained model to your workspace in a file called
    # trainedmodel.pkl
    outputpath = root + '/' + model_path
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    model_name = 'trainedmodel.pkl'
    logger.info(f'Saving model {os.path.join(model_path, model_name)}')
    filehandler = open(outputpath + '/' + model_name, 'wb')
    pickle.dump(lr, filehandler)


train_model()