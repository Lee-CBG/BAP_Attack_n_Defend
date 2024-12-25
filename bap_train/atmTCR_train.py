import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import deque
from typing import Union
import logging
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score


import torch
import torch.nn.functional as F
import torch.optim as optim

# set all models in the same level of directory as bap_attack repo
# not sure if this is already appended in the main script. TODO: should not.
"""Script might start at different level based on the debugger setting.
Fix prefix at 'attack' level to accomendate all plugin reward repos.
"""


# TODO: Try to rule out dependency in the future version. Need reward model dependency now.
cwd = os.getcwd()
root_dir = 'attack'
prefix = cwd[:(cwd.find(root_dir)+len(root_dir))] if cwd.find(root_dir)!=-1	 else cwd+f'/{root_dir}' # in case cwd is below root_dir level

REWARD_DIR = Path(prefix).joinpath('ATM-TCR')
sys.path.append(str(REWARD_DIR))

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
DATASET_DIR = Path(prefix).joinpath('bap_attack/data/tcr_split')
OUTPUT_DIR =  Path(prefix).joinpath('bap_attack')
# NEW_MODEL = 'model_list/atmTCR_retrain_1.ckpt'
# OLD_MODEL = 'model_list/atmTCR_retrain.ckpt'
NEW_MODEL = 'model_list/atmTCR_tcr_retrain.ckpt'
OLD_MODEL = ''


from data_loader import define_dataloader, load_embedding
from attention import Net
from utils import get_performance_batchiter


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
	filename = Path(prefix).joinpath(f'bap_attack/logs/{Path(NEW_MODEL).stem}.log')
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=Path, default=DATASET_DIR)
parser.add_argument('-o', '--output_dir', type=Path, default=OUTPUT_DIR)
parser.add_argument('--new_model', type=str, default=NEW_MODEL)
parser.add_argument('--old_model', type=str, default=OLD_MODEL)
parser.add_argument('--epochs', type=int, default=200)


@dataclass
class Args:
	blosum: Union[str, None] = str(REWARD_DIR.joinpath('data/blosum/BLOSUM45'))
	lin_size: int = 1024
	max_len_pep: int = 22
	max_len_tcr: int = 20
	drop_rate: float = 0.25
	heads: int = 5
	padding: str = 'mid'
	batch_size: int = 128
	lr: float = 0.00001
	early_stop: bool = False
	min_epoch: int = 30
	shuffle = True


def train(model, train_loader, optimizer):
    model.train()
    for batch in train_loader:
        x_pep, x_tcr, y = batch.X_pep.to(
            DEVICE), batch.X_tcr.to(DEVICE), batch.y.to(DEVICE)
        optimizer.zero_grad()
        yhat = model(x_pep, x_tcr)
        y = y.unsqueeze(-1).expand_as(yhat)
        loss = F.binary_cross_entropy(yhat, y)
        loss.backward()
        optimizer.step()

def load_dataset(dataDir, dataset, shuffle, partial):
	pickle_file = Path(dataDir).joinpath(dataset)
	df = pd.read_pickle(pickle_file)
	l = df.shape[0]
	threshold = int(l * partial)
	
	# return array of peptides, array of tcrs, array of labels
	if partial <= 0:
		x_pep = df.iloc[threshold-1:l]['epi'].values
		x_tcr = df.iloc[threshold-1:l]['tcr'].values
		y = df.iloc[threshold-1:l]['binding'].values
	else:
		x_pep =df.iloc[0:threshold]['epi'].values
		x_tcr = df.iloc[0:threshold]['tcr'].values
		y = df.iloc[0:threshold]['binding'].values
	index = np.arange(x_pep.shape[0])
	if shuffle:
		np.random.shuffle(index)		
	return x_pep[index], x_tcr[index], y[index]

def shuffle_data(x_pep, x_tcr, y):
	index = np.arange(x_pep.shape[0])
	np.random.shuffle(index)
	return x_pep[index], x_tcr[index], y[index]	


def atm_tcr(dataset_dir, output_dir, new_model, old_model, epochs):
	if not old_model:
		old_model = None
	else:
		Args.lr *= 0.01

	embedding_matrix = load_embedding(filename=str(Args.blosum))
	model = Net(embedding_matrix, Args).to(DEVICE)
	if old_model is not None:
		model.load_state_dict(torch.load(output_dir.joinpath(old_model), map_location=DEVICE))
		logger.info(f'Model loaded from {output_dir.joinpath(old_model)}')
		train_epi, train_tcr, train_label = load_dataset(dataset_dir, 'training.pkl', shuffle=True, partial=0.8)
		val_epi, val_tcr, val_label = load_dataset(dataset_dir, 'training.pkl', shuffle=False, partial=-0.2)
		train_epi, train_tcr, train_label = load_dataset(dataset_dir, 'trainData.pkl', shuffle=True, partial=0.8)
		val_epi, val_tcr, val_label = load_dataset(dataset_dir, 'trainData.pkl', shuffle=False, partial=-0.2)
	else:
		train_epi, train_tcr, train_label = load_dataset(dataset_dir, 'training.pkl', shuffle=True, partial=0.8)
		val_epi, val_tcr, val_label = load_dataset(dataset_dir, 'training.pkl', shuffle=False, partial=-0.2)
	
	logger.info(f'Loading training dataset from {dataset_dir}')
	train_loader = define_dataloader(train_epi, train_tcr, train_label,
                                     Args.max_len_pep, Args.max_len_tcr,
                                     padding=Args.padding,
                                     batch_size=Args.batch_size, device=DEVICE)
	val_loader = define_dataloader(val_epi, val_tcr, val_label,
									Args.max_len_pep, Args.max_len_tcr,
									padding=Args.padding,
									batch_size=Args.batch_size, device=DEVICE)
	
	optimizer = optim.Adam(model.parameters(), lr=Args.lr)
	lossArraySize = 10
	lossArray = deque([sys.maxsize], maxlen=lossArraySize)
	auc = []
	accuracy = []
	for epoch in tqdm(range(epochs)):
		train(model, train_loader['loader'], optimizer)
		perf_test = get_performance_batchiter(val_loader['loader'], model, DEVICE)
		lossArray.append(perf_test['loss'])
		auc.append(perf_test['auc'])
		accuracy.append(perf_test['accuracy'])
		if len(lossArray) > 1:
			average_loss_change = sum(np.abs(np.diff(lossArray))) / (len(lossArray) - 1)
		else:
			average_loss_change = float('inf')
		if epoch % 2 == 0 and epoch > 0:
			torch.save(model.state_dict(), output_dir.joinpath(new_model))
			logger.info(f'Epoch {epoch} - AUC: {auc[-1]} - Accuracy: {accuracy[-1]} - Loss: {lossArray}')
		if epoch > Args.min_epoch and average_loss_change < 10 and Args.early_stop:
			logger.info(f"Early stopping at epoch {epoch}")
			# chamge the model name to the checkpoint
			checkpoint = f'{Path(new_model).stem}_checkpoint.ckpt'
			torch.save(model.state_dict(), output_dir.joinpath(checkpoint))
			# torch.save(model.state_dict(), output_dir.joinpath(new_model))
			break
		train_epi, train_tcr, train_label = shuffle_data(train_epi, train_tcr, train_label)

	torch.save(model.state_dict(), output_dir.joinpath(new_model))
	
	if old_model is None:
		test_epi, test_tcr, test_label = load_dataset(dataset_dir, 'testing.pkl', shuffle=False, partial=1)
	else:
		test_epi, test_tcr, test_label = load_dataset(dataset_dir, 'testData.pkl', shuffle=False, partial=1)
	test_loader = define_dataloader(test_epi, test_tcr, test_label, 
	                            maxlen_pep=Args.max_len_pep,
	                            maxlen_tcr=Args.max_len_tcr,
	                            padding=Args.padding,
                                shuffle=Args.shuffle,
	                            batch_size=Args.batch_size,
	                            device=DEVICE)
	logger.info(f'Loading testing dataset from {dataset_dir}')
	yhat_list = []
	y_list = []
	for batch in test_loader['loader']:
		X_pep, X_tcr, y = batch.X_pep.to(DEVICE), batch.X_tcr.to(DEVICE), batch.y.to(DEVICE)
		yhat_list.append(model(X_pep, X_tcr).cpu().detach().numpy())
		y_list.append(y)
	yhat = np.vstack(yhat_list)
	y = np.vstack(y_list)
	roc = roc_auc_score(y, yhat)
	logger.info(f'AUC: {roc}')
	acc = accuracy_score(y, np.round(yhat))
	logger.info(f'AUC: {acc}')


if __name__ == '__main__':
	args = parser.parse_args()
	atm_tcr(args.dataset, args.output_dir, args.new_model, args.old_model, args.epochs)