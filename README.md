# Background

Imagine that you're the Chief Data Scientist at a big company that has 10,000 corporate clients. Your company is extremely concerned about attrition risk: the risk that some of their clients will exit their contracts and decrease the company's revenue. They have a team of client managers who stay in contact with clients and try to convince them not to exit their contracts. However, the client management team is small, and they're not able to stay in close contact with all 10,000 clients.

The company needs you to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's 10,000 clients. If the model you create and deploy is accurate, it will enable the client managers to contact the clients with the highest risk and avoid losing clients and revenue.

Creating and deploying the model isn't the end of your work, though. Your industry is dynamic and constantly changing, and a model that was created a year or a month ago might not still be accurate today. Because of this, you need to set up regular monitoring of your model to ensure that it remains accurate and up-to-date. You'll set up processes and scripts to re-train, re-deploy, monitor, and report on your ML model, so that your company can get risk assessments that are as accurate as possible and minimize client attrition.


![Project: a dynamic risk assessment system](front.png)

## Dependencies
All the dependencies are listed in the `requirements.txt` file. You can setup a virtual environment using [Anaconda](https://www.anaconda.com/products/distribution) and install the required dependencies there.

## Setup

```bash
> conda env create -f environment.yml
> conda activate riskass
```

If environment needs modifications, reinstall after:
```bash
> conda env remove -n riskass
```

## Data
The data represents records of corporations, their characteristics, and their historical attrition records. 
One row represents a hypothetical corporation. There are five columns in the dataset:

* "corporation", which contains four-character abbreviations for names of corporations
* "lastmonth_activity", which contains the level of activity associated with each corporation over the previous month
* "lastyear_activity", which contains the level of activity associated with each corporation over the previous year
* "number_of_employees", which contains the number of employees who work for the corporation
* "exited", which contains a record of whether the corporation exited their contract (1 indicates that the corporation exited, and 0 indicates that the corporation did not exit)

The dataset's final column, "exited", is the target variable for our predictions.

## Steps

You'll complete the project by proceeding through 5 steps:

1. Data ingestion. Automatically check a database for new data that can be used for model training. Compile all training data to a training dataset and save it to persistent storage. Write metrics related to the completed data ingestion tasks to persistent storage.
2. Training, scoring, and deploying. Write scripts that train an ML model that predicts attrition risk, and score the model. Write the model and the scoring metrics to persistent storage.
3. Diagnostics. Determine and save summary statistics related to a dataset. Time the performance of model training and scoring scripts. Check for dependency changes and package updates.
4. Reporting. Automatically generate plots and documents that report on model metrics. Provide an API endpoint that can return model predictions and metrics.
5. Process Automation. Create a script and cron job that automatically run all previous steps at regular intervals.

### Step 1: Data ingestion
We will be using `ingestion.py` for the python code and `config.json` as a configuration file. 
The `config.json` contains five entries:
* `input_folder_path`, which specifies the location where your project will look for input data, to ingest,
and to use in model training. 
* `output_folder_path`, which specifies the location to store output files related to data ingestion. 
* `test_data_path`, which specifies the location of the test dataset
* `output_model_path`, which specifies the location to store the trained models and scores.
* `prod_deployment_path`, which specifies the location to store the models in production.

### Step 2: Training, scoring and deploying
We will be using `training.py` to train a logistic regression model. 
Furthermore `scoring.py` and `deployment.py` will evaluate the model and save it and stats in deployment folder. 

### Step 3: Model and data diagnostics
`diagnostics.py` runs inference of the model on the testdata.
It also does data checks: summary of statistics (mean, median, std) and checking for nans.
Furthermore, the execution time of training and ingestion is done and dependencies are checked on their versions
and compared with the latest versions available. 

### Step 4: Reporting
`reporting.py` saves a confusion matrix plot to disk. It shows the number of FP, TP, FN, TN obtained with our model.
Furthermore, the `app.py` can be called:
```bash
python app.py
```
which spins up an api on localhost. It can be accessed in your browser at [link](http://127.0.0.1:8000/).
But, we call also call `apicalls.py`:
```bash
python apicalls.py
```
This will make requests for following endpoints:
* [prediction](http://127.0.0.1:8000/prediction?filename=testdata/testdata.csv)
This endpoint should take a dataset's file location as its input, and return the outputs of the prediction function you created in Step 3.
* [scoring](http://127.0.0.1:8000/scoring)
This endpoint needs to run the scoring.py script you created in Step 2 and return its output.
* [summary](http://127.0.0.1:8000/summary)
This endpoint needs to run the summary statistics function you created in Step 3 and return its outputs.
* [diagnostics](http://127.0.0.1:8000/diagnostics)
This endpoint needs to run the timing, missing data, and dependency check functions you created in Step 3 and return their outputs.

## Starter Files
There are many files in the starter: 10 Python scripts, one configuration file, one requirements file, and five datasets.

The following are the Python files that are in the starter files:

* training.py, a Python script meant to train an ML model
* scoring.py, a Python script meant to score an ML model
* deployment.py, a Python script meant to deploy a trained ML model
* ingestion.py, a Python script meant to ingest new data
* diagnostics.py, a Python script meant to measure model and data diagnostics
* reporting.py, a Python script meant to generate reports about model metrics
* app.py, a Python script meant to contain API endpoints
* wsgi.py, a Python script to help with API deployment
* apicalls.py, a Python script meant to call your API endpoints
* fullprocess.py, a script meant to determine whether a model needs to be re-* deployed, and to call all other Python scripts when needed
The following are the datasets that are included in your starter files. Each of them is fabricated datasets that have information about hypothetical corporations.

Note: these data have been uploaded to your workspace as well

* dataset1.csv and dataset2.csv, found in /practicedata/
* dataset3.csv and dataset4.csv, found in /sourcedata/
* testdata.csv, found in /testdata/
The following are other files that are included in your starter files:

* requirements.txt, a text file and records the current versions of all the modules that your scripts use
* config.json, a data file that contains names of files that will be used for configuration of your ML Python scripts

## Misscelanious

```bash
> autopep8 --in-place --aggressive --aggressive [script_name].py
```
