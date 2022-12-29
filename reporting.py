import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


from diagnostics import model_predictions


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

root = os.getcwd()
dataset_csv_path = os.path.join(root, config['output_folder_path'])
test_data_path = os.path.join(root, config['test_data_path'])
output_model_path = os.path.join(root, config['output_model_path'])


# Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    X = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    y = X['exited']
    X.drop(['corporation', 'exited'], inplace=True, axis=1)

    # we should ensure dataset in model_predictions() is the same
    y_pred = model_predictions(X)

    labels = [0, 1]
    cm = metrics.confusion_matrix(y, y_pred, labels=labels)
    cmd = metrics.ConfusionMatrixDisplay(cm)
    cmd.plot()
    cmd.figure_.savefig(
        os.path.join(
            os.getcwd(),
            output_model_path,
            'confusionmatrix.png'),
        dpi=300)


if __name__ == '__main__':
    score_model()
