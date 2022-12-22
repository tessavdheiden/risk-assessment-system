from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

output_folder_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']
output_model_path = config['output_model_path']


# function for deployment
def store_model_into_pickle():
    # copy the latest pickle file, the latestscore.txt value, and the
    # ingestfiles.txt file into the deployment directory
    root = os.getcwd()

    # load model
    inputmodelpath = os.path.join(root, output_model_path, 'trainedmodel.pkl')
    with open(inputmodelpath, 'rb') as file:
        model = pickle.load(file)
    # load scores
    outputscorepath = os.path.join(root, output_model_path, 'latestscore.txt')
    scores = open(outputscorepath)
    # load data
    ingestfilespath = os.path.join(
        root, output_folder_path, 'ingestedfiles.txt')
    ingestfiles = open(ingestfilespath)

    if not os.path.exists(root + '/' + prod_deployment_path):
        os.makedirs(root + '/' + prod_deployment_path)

    # save model
    outputmodelpath = os.path.join(
        root, prod_deployment_path, 'trainedmodel.pkl')
    filehandler = open(outputmodelpath, 'wb')
    pickle.dump(model, filehandler)

    # save scores
    scorepath = os.path.join(root, prod_deployment_path, 'latestscore.txt')
    with open(scorepath, 'w') as b:
        for line in scores:
            b.write(line)

    ingestpath = os.path.join(root, prod_deployment_path, 'ingestedfiles.txt')
    with open(ingestpath, 'w') as b:
        for line in ingestfiles:
            b.write(line)


store_model_into_pickle()
