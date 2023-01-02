import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging
from sqlalchemy import create_engine
import glob


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


# Function for data ingestion
def merge_multiple_dataframe(engine):
    # check for datasets, compile them together, and write to an output file
    root = os.getcwd()

    datasets = glob.glob(f'{root}/{input_folder_path}/*.csv')
    final_dataframe = pd.concat(map(pd.read_csv, datasets))

    # drop duplicates
    final_dataframe = final_dataframe.drop_duplicates()
    # save file
    outputpath = root + '/' + output_folder_path
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    # save data
    targetpath = outputpath + '/' + 'finaldata.csv'
    try:
        final_dataframe.to_csv(targetpath, index=False)
        logger.info(f'File saved to {"/".join(targetpath.split("/")[-2:])}')
    except BaseException:
        logger.info(f'Failed to save {"/".join(targetpath.split("/")[-2:])}')

    # save records
    targetpath = outputpath + '/' + 'ingestedfiles.txt'
    with open(targetpath, 'w') as f:
        for name in datasets:
            f.write(name + '\n')

    final_dataframe.to_sql(con=engine, name='Ingestion', if_exists='replace')


if __name__ == '__main__':
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                           .format(host="localhost", db="riskassessment", user="root", pw="Mohammed123..."))
    merge_multiple_dataframe(engine)
