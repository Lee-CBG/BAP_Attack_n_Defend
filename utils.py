import os
import csv
import pandas as pd

def extract_epis_tcrs(dataset):
    # Split the dataset by the delimiter '$'
    parts = dataset.split('$')
    
    # Extract epitopes and T-cell receptors
    epis = ''.join(parts[0].strip().split())
    tcrs = ''.join(parts[1].strip().replace('<EOS>', '').split())
    
    return epis, tcrs


def save_to_csv(datasets, csv_file_path='tmp_epis_tcrs.csv'):
    # Initialize lists to store extracted epis and tcrs
    all_epis = []
    all_tcrs = []

    # Extract epis and tcrs from each dataset
    for dataset in datasets:
        epis, tcrs = extract_epis_tcrs(dataset)
        all_epis.append(epis)
        all_tcrs.append(tcrs)

    # Create a list of dictionaries for each pair of epis and tcrs
    data = [{'Epitopes': epis, 'TCRs': tcrs} for epis, tcrs in zip(all_epis, all_tcrs)]

    # Write the data to a CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        fieldnames = ['Epitopes', 'TCRs']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)
            
            
def save_to_csv_1(epis, tcrs, csv_file_path='tmp_epis_tcrs.csv'):

    # Create a list of dictionaries for each pair of epis and tcrs
    data = [{'Epitopes': epi, 'TCRs': tcr} for epi, tcr in zip(epis, tcrs)]

    # Write the data to a CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        fieldnames = ['Epitopes', 'TCRs']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)
            
            
            
def append_tmp_to_master(tmp_filename='tmp_epis_tcrs.csv', master_filename='all_results.csv'):

    master_exists = os.path.isfile(master_filename)
    tmp_df = pd.read_csv(tmp_filename)
    tmp_df.to_csv(master_filename, mode='a', header=not master_exists, index=False)