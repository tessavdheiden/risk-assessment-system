from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
import diagnostics
import scoring
# import predict_exited_from_saved_model
import json
import os


# Set up variables for use in our script
app = Flask(__name__)
# app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

prediction_model = None


def readpandas(filename):
    thedata = pd.read_csv(filename)
    return thedata

@app.route('/')
def index():
    return 'Welcome!'

# Prediction Endpoint

@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    # call the prediction function you created in Step 3
    filename = request.args.get('filename')
    X = pd.read_csv(filename)
    X.drop(['corporation', 'exited'], inplace=True, axis=1)

    predictions = diagnostics.model_predictions(X)
    # add return value for prediction outputs
    return {'predictions': str(predictions)}

# #######################Scoring Endpoint


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def stats_score():
    # check the score of the deployed model
    score = scoring.score_model()
    # add return value (a single F1 score number)
    return {'score': f'{score:.2f}'}

# Summary Statistics Endpoint


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats_summary():
    # check means, medians, and modes for each column
    dfsummary = diagnostics.dataframe_summary()
    dfsummary = [f'{stat:.2f}' for stat in dfsummary]
    # return a list of all calculated summary statistics
    return {'summary': str(dfsummary)}

# #######################Diagnostics Endpoint


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def stats_diagnostics():
    # check timing and percent NA values
    [training_time, ingestion_time] = diagnostics.execution_time()
    missing_data = diagnostics.missing_data()
    missing_data = [f'{stat:.2f}' for stat in missing_data]
    outdated_package_list = diagnostics.outdated_packages_list()
    result = {
        'training_time': f'{training_time:.2f}',
        'ingestion_time': f'{ingestion_time:.2f}',
        'missing_data': missing_data,
    }
    return result  # add return value for all diagnostics


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
