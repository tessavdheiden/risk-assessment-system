import os
import requests
import json
import pickle

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"


# Call each API endpoint and store the responses
response1 = requests.post(
    URL +
    f'/prediction?filename=testdata/testdata.csv')  # put an API call here
response2 = requests.get(URL + f'/scoring')  # put an API call here
response3 = requests.get(URL + f'/summarystats')  # put an API call here
response4 = requests.get(URL + f'/diagnostics')  # put an API call here

# #combine all API responses
responses = [
    response1,
    response2,
    response3,
    response4]  # combine reponses here

for response in responses:
    assert response.status_code == 200

# write the responses to your workspace
with open('config.json', 'r') as f:
    config = json.load(f)

root = os.getcwd()
output_model_path = os.path.join(root, config['output_model_path'])
with open(os.path.join(output_model_path, 'apireturns.txt'), "w") as f:
    for response in responses:
        f.write(str(response.json()))
