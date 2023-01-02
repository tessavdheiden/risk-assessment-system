from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


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
    outputmodelpath = os.path.join(root, prod_deployment_path, 'trainedmodel.pkl')
    shutil.copy2(inputmodelpath, outputmodelpath)
    logger.info(f'Copied model from {inputmodelpath} to {outputmodelpath}')

    # load scores and ingestfiles
    inputscorepath = os.path.join(root, output_model_path, 'latestscore.txt')
    outputscorepath = os.path.join(root, prod_deployment_path, 'latestscore.txt')
    shutil.copy2(inputscorepath, outputscorepath)
    logger.info(f'Copied score from {inputscorepath} to {outputscorepath}')

    inputingestfilespath = os.path.join(
        root, output_folder_path, 'ingestedfiles.txt')
    outputingestfilespath = os.path.join(root, prod_deployment_path, 'ingestedfiles.txt')
    shutil.copy2(inputingestfilespath, outputingestfilespath)
    logger.info(f'Copied score from {inputingestfilespath} to {outputingestfilespath}')


if __name__ == '__main__':
    store_model_into_pickle()
