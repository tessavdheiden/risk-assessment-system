import ast
import os
import json
import sys
import logging
import pickle
import pandas as pd
from sklearn import metrics
import subprocess


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


root = os.getcwd()
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
prod_deployment_path = config['prod_deployment_path']
output_folder_path = config['output_folder_path']


'''
Check and read new data
'''
# first, read ingestedfiles.txt
with open(os.path.join(root, prod_deployment_path, 'ingestedfiles.txt'), 'r') as f:
    # create list from txt entries
    old_ingested_files = f.read().replace("\n", " ").split()

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
new_ingested_files = [
    x for x in os.listdir(
        os.path.join(
            root,
            input_folder_path)) if '.csv' in x]

new_files = list(set(new_ingested_files) - set(old_ingested_files))
if len(new_files) > 0:
    os.system('python3 ingestion.py')

'''
Deciding whether to proceed, part 1
'''
# if you found new data, you should proceed. otherwise, do end the process here
if len(new_files) == 0:
    sys.exit()

'''
Checking for model drift
'''
# check whether the score from the deployed model is different from the score
# from the model that uses the newest ingested data
# 1. Read the score from the latest model, recorded in latestscore.txt
# from the deployment directory, specified in the prod_deployment_path entry of your config.json file.
with open(os.path.join(root, prod_deployment_path, 'latestscore.txt'), 'r') as f:
    deployed_model_score = float(f.read())
# 2. Make predictions using the trainedmodel.pkl model in the /production_deployment directory
# and the most recent data you obtained from the previous "Checking and Reading New Data" step.


def score_model(model_path, data_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    X = pd.read_csv(data_path)
    X.drop('corporation', inplace=True, axis=1)
    y = X.pop('exited')

    predicted = model.predict(X)
    f1score = metrics.f1_score(predicted, y)
    return f1score


# 3. Get a score for the new predictions from step 2 by running the scoring.py.
new_model_score = score_model(
    model_path=os.path.join(root, prod_deployment_path, 'trainedmodel.pkl'),
    data_path=os.path.join(root, output_folder_path, 'finaldata.csv')
)
os.system('python3 scoring.py')

# 4. Check whether the new score from step 3 is higher or lower than the score recorded in latestscore.txt in
# step 1 using the raw comparison test.
# If the score from step 3 is lower, then model drift has occurred. Otherwise, it has not.
model_drift = new_model_score < deployed_model_score

'''
Deciding whether to proceed, part 2
'''
# if you found model drift, you should proceed. otherwise, do end the process here
if not model_drift:
    sys.exit()
else:
    os.system('python3 training.py')
'''
Re-deployment
'''
# if you found evidence for model drift, re-run the deployment.py script
if model_drift:
    os.system('python3 deployment.py')

'''
Diagnostics and reporting
'''
# run diagnostics.py and reporting.py for the re-deployed model
if model_drift:
    os.system('python3 reporting.py')
    response = subprocess.run(['curl', '127.0.0.1:8000'], capture_output=True).stdout
    if response.decode("utf-8") == 'Welcome!':
        os.system('python3 apicalls.py')

