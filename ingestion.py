import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


# Function for data ingestion
def merge_multiple_dataframe():
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
            namelist.append(fullpath)
            try:
                final_dataframe = final_dataframe.append(
                    currentdf).reset_index(drop=True)
            except BaseException:
                logger.info(
                    f'Cannot append dataframe with columns {currentdf.columns}')
                return False
        except BaseException:
            logger.info(f'Failed to read {fullpath}')
            pass

    # drop duplicates
    final_dataframe = final_dataframe.drop_duplicates()
    # save file
    outputpath = root + '/' + output_folder_path
    if not os.path.exists(outputpath):
        print(os.path.isfile(outputpath))
        os.makedirs(outputpath)

    # save data
    targetpath = outputpath + '/' + 'finaldata.csv'
    try:
        final_dataframe.to_csv(targetpath, index=False)
        logger.info(f'File saved to {targetpath}')
    except BaseException:
        logger.info(f'Failed to save {targetpath}')

    # save records
    targetpath = outputpath + '/' + 'ingestedfiles.txt'
    with open(targetpath, 'w') as f:
        for name in namelist:
            f.write(name + '\n')


if __name__ == '__main__':
    merge_multiple_dataframe()
