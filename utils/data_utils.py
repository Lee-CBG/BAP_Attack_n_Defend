import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
SEED = 42

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
		data.to_csv(csv_file_path, header= True, index=False, mode='w', float_format='%.5f')
	else:
		data.to_csv(csv_file_path, header= False, index=False, mode='a', float_format='%.5f')

def override_csv(info, rd, csv_file_path='attack.csv'):
	data = pd.DataFrame(info)
	data['Iter'] = rd
	# Create a list of dictionaries for each pair of epis and tcrs
	# Write the data to a CSV file
	if not os.path.exists(csv_file_path.parent):
		Path.mkdir(csv_file_path.parent, parents=True)
	data.to_csv(csv_file_path, header= True, index=False, mode='w', float_format='%.5f')

def merge_results(tmp_filename, master_filename=None):
	contents = pd.read_csv(tmp_filename)
	if not master_filename:
		master_filename = Path(tmp_filename).parent.joinpath(Path(tmp_filename).stem+'_output.csv')
	if not os.path.exists(master_filename):
		contents.to_csv(master_filename, header=True, index=False, mode='w', float_format='%.5f')
	else:
		contents.to_csv(master_filename, header=False, index=False, mode='a', float_format='%.5f')

def select_candidates(file, method):
	storage_dir = Path(file).parent
	contents = pd.read_csv(file)
	# try:
	# 	Path.mkdir(storage_path, parents=True, exist_ok=False)
	# except FileExistsError:
		# print(f'Folder {storage_dir} already exists')
		# exit(1)
	if method == 'neg_control':
		candidates = contents[(contents['yhat'] > 0) & 
						(contents['gpt_ll_mean'] < 1.06) & 
						(contents['tcr_match'] < 0.94)]
		candidates = candidates.reset_index(drop=True)
		candidates = candidates.drop(columns=['yhat', 'gpt_ll', 'gpt_ll_mean', 'tcr_match'])
		candidates['binding'] = 0
		candidates['groups'] = 'neg_control'
		#  TODO: rename tcr and epi
		candidates.to_csv(storage_dir.joinpath('neg_control.csv'), header=True, index=False)
	else:
		raise NameError('Method not recognized')
	# return storage_dir

def augment_dataset(attack, storage_path, original_data):
	if attack in ['pite', 'catelmp-mlp']:
		# TODO: construct sampling method for .npy and csv
		neg_control = pd.read_pickle(storage_path.joinpath('neg_control.pkl'))
		neg_control = neg_control.rename(columns={'Epitopes': 'epi', 'TCRs': 'tcr'})
		neg_control['tcr_embeds'] = neg_control['tcr_embeds'].apply(lambda x: np.array(x))
		neg_control['epi_embeds'] = neg_control['epi_embeds'].apply(lambda x: np.array(x))
		n = len(neg_control)
		pool = '/mnt/disk11/user/pzhang84/data/npzs'
		disk_mapping = '/mnt/disk11/user/pzhang84/data/mappings.csv'
		idx = pd.read_csv(disk_mapping)
		selected_rows = pd.concat([
			idx[idx['groups'] == 'neg_healthy'].sample(n=n, random_state=SEED),
			idx[idx['groups'] == 'neg_shuffle'].sample(n=n, random_state=SEED),
			idx[idx['groups'] == 'pos'].sample(n=3*n, random_state=SEED)
			], ignore_index=True)
				 #  .reset_index(drop=True)
		# load few npz files from pool
		selected_rows['tcr_embeds'] = 0
		selected_rows['epi_embeds'] = 0
		tcr = []
		epi = []
		for i in selected_rows.index:
			idx = selected_rows.loc[i, 'npz_id'] - 1
			contents = np.load((f'/mnt/disk11/user/pzhang84/data/npzs/{idx}.npz'))
			tcr.append(contents['x_tcr'])
			epi.append(contents['x_epi'])
		selected_rows['tcr_embeds'] = tcr
		selected_rows['epi_embeds'] = epi
		selected_rows.reset_index(drop=True)
	else:
		neg_control = pd.read_csv(storage_path.joinpath('neg_control.csv'))
		neg_control = neg_control.rename(columns={'Epitopes': 'epi', 'TCRs': 'tcr'})
		# neg_control['tcr_embeds'] = neg_control['tcr_embeds'].apply(lambda x: np.array(x))
		# neg_control['epi_embeds'] = neg_control['epi_embeds'].apply(lambda x: np.array(x))
		n = len(neg_control)
		pool = pd.read_pickle(original_data.joinpath('pool.pkl'))
		# update data is limited, so only consider the pickled data
		selected_rows = pd.concat([
			pool[pool['groups'] == 'neg_healthy'].sample(n=n, random_state=SEED),
			pool[pool['groups'] == 'neg_shuffle'].sample(n=n, random_state=SEED),
 			pool[pool['groups'] == 'pos'].sample(n=3*n, random_state=SEED)
			], ignore_index=True)
		selected_rows.drop(columns=['tcr_embeds', 'epi_embeds'], inplace=True)
	combined_df = pd.concat([selected_rows, neg_control]).reset_index(drop=True)
	train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=SEED)
	train_df.to_pickle(f'{storage_path}/trainData.pkl')
	test_df.to_pickle(f'{storage_path}/testData.pkl')

def clean_result(dir='experiments'):
	# only clean result files generated by the experiment runner
	# if not dir:
	# 	dir = 'experiments'
	if dir != 'experiments':
		input('Are you sure you want to delete the folder?')
	assert('attack' in str(dir)), NameError('Unsafe operation')
	folders = os.listdir(dir)
	for f in folders:
		if 'iter' not in f or f not in ['tmp', 'result']:
			raise NameError('Unsafe operation')
	shutil.rmtree(dir) 
	
