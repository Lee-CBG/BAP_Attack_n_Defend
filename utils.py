import os
import csv
import pandas as pd
import numpy as np
from pathlib import Path


def extract_epis_tcrs(dataset):
    # Split the dataset by the delimiter '$'
    parts = dataset.split('$')

    # Extract epitopes and T-cell receptors
    epis = ''.join(parts[0].strip().split())
    tcrs = ''.join(parts[1].strip().replace('<EOS>', '').split())
    
    return epis, tcrs

def save_to_csv(info, round, csv_file_path='attack.csv'):
    data = pd.DataFrame(info)
    data['Iter'] = round
    # Create a list of dictionaries for each pair of epis and tcrs
    # Write the data to a CSV file
    if not os.path.exists(csv_file_path.parent):
        os.makedirs(csv_file_path.parent,  exist_ok=True)
    if not os.path.exists(csv_file_path):
        data.to_csv(csv_file_path, header= True, index=False, mode='w')
    else:
        data.to_csv(csv_file_path, header= False, index=False, mode='a')
        
def override_csv(info, rd, csv_file_path='attack.csv'):
    data = pd.DataFrame(info)
    data['Iter'] = rd
    # Create a list of dictionaries for each pair of epis and tcrs
    # Write the data to a CSV file
    if not os.path.exists(csv_file_path.parent):
        os.mkdir(csv_file_path.parent)
    data.to_csv(csv_file_path, header= True, index=False, mode='w')


def append_tmp_to_master(tmp_filename='tmp_epis_tcrs.csv', master_filename='all_results.csv'):

    master_exists = os.path.isfile(master_filename)
    tmp_df = pd.read_csv(tmp_filename)
    tmp_df.to_csv(master_filename, mode='a', header=not master_exists, index=False)
    

    
