import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging
from sqlalchemy import create_engine


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
    final_dataframe = pd.DataFrame(
        columns=[
            'corporation',
            'lastmonth_activity',
            'lastyear_activity',
            'number_of_employees',
            'exited'
        ]
    )
    root = os.getcwd()
    filenames = os.listdir(root + '/' + input_folder_path)
    namelist = []
    for each_filename in filenames:
        fullpath = root + '/' + input_folder_path + '/' + each_filename
        try:
            currentdf = pd.read_csv(fullpath)
            namelist.append(each_filename)
            try:
                final_dataframe = final_dataframe.append(
                    currentdf).reset_index(drop=True)
            except BaseException:
                logger.info(
                    f'Cannot append dataframe with columns {currentdf.columns}')
                return False
        except BaseException:
            logger.info(f'Failed to read {each_filename}')
            pass

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
        for name in namelist:
            f.write(name + '\n')

    final_dataframe.to_sql(con=engine, name='Ingestion', if_exists='replace')


if __name__ == '__main__':
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                           .format(host="localhost", db="riskassessment", user="root", pw="Mohammed123..."))
    merge_multiple_dataframe(engine)
