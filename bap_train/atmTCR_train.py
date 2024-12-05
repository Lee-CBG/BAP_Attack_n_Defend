import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from dataclasses import dataclass
from collections import deque
from typing import Union
from tqdm import tqdm
import argparse
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
PRINT_EVERY_EPOCH = 1
DATASET_DIR = Path(prefix).joinpath('bap_attack/data/tcr_split')
OUTPUT_DIR =  Path(prefix).joinpath('bap_attack/model_list')
NEW_MODEL = 'atmTCR_retrain.ckpt'

from data_loader import define_dataloader, load_embedding
from attention import Net
from utils import get_performance_batchiter, get_performance


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=Path, default=DATASET_DIR)
parser.add_argument('-o', '--output_dir', type=Path, default=OUTPUT_DIR)
parser.add_argument('--new_model', type=str, default=NEW_MODEL)
parser.add_argument('--old_model', type=Union[str, None], default=None)
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
	batch_size: int = 32
	lr: float = 0.001
	early_stop: bool = False
	min_epoch: int = 30



def train(model, device, train_loader, optimizer, epoch):

    model.train()

    for batch in train_loader:

        x_pep, x_tcr, y = batch.X_pep.to(
            device), batch.X_tcr.to(device), batch.y.to(device)

        optimizer.zero_grad()
        yhat = model(x_pep, x_tcr)
        y = y.unsqueeze(-1).expand_as(yhat)
        loss = F.binary_cross_entropy(yhat, y)
        loss.backward()
        optimizer.step()

    if epoch % PRINT_EVERY_EPOCH == 1:
        print('[TRAIN] Epoch {} Loss {:.4f}'.format(epoch, loss.item()))


def load_dataset(pickle_file):
	df = pd.read_pickle(pickle_file)
	# return array of peptides, array of tcrs, array of labels
	x_pep = df['epi'].values
	x_tcr = df['tcr'].values
	y = df['binding'].values
	return x_pep, x_tcr, y

def atm_tcr(dataset_dir, output_dir, new_model, old_model, epochs):
	embedding_matrix = load_embedding(filename=str(Args.blosum))
	model = Net(embedding_matrix, Args).to(DEVICE)
	if old_model is not None:
		try: 
			# Try to load the model from re-trained model path
			model.load_state_dict(torch.load(output_dir.joinpath(old_model), map_location=DEVICE))
			print(f'Model loaded from {output_dir.joinpath(old_model)}')
		except Exception as e:
			print(e)
			print(f"Model not retrained yet.")
			sys.exit(1)
	else:
		print('Retrain bap model...\n')
	# BAP model update
	train_epi, train_tcr, train_label = load_dataset(dataset_dir.joinpath('training.pkl'))
	print(f'Loading traing dataset from {dataset_dir}')
	test_epi, test_tcr, test_label = load_dataset(dataset_dir.joinpath('testing.pkl'))
	print(f'Loading testing dataset from {dataset_dir}')
	train_loader = define_dataloader(train_epi, train_tcr, train_label,
                                     Args.max_len_pep, Args.max_len_tcr,
                                     padding=Args.padding,
                                     batch_size=Args.batch_size, device=DEVICE)
	test_loader = define_dataloader(test_epi, test_tcr, test_label,
									Args.max_len_pep, Args.max_len_tcr,
									padding=Args.padding,
									batch_size=Args.batch_size, device=DEVICE)
	optimizer = optim.Adam(model.parameters(), lr=Args.lr)
	lossArraySize = 10
	lossArray = deque([sys.maxsize], maxlen=lossArraySize)
	for e in tqdm(range(epochs)):
		train(model, DEVICE, train_loader['loader'], optimizer, e)
		perf_test = get_performance_batchiter(test_loader['loader'], model, DEVICE)
		lossArray.append(perf_test['loss'])
		average_loss_change = sum(np.abs(np.diff(lossArray))) / lossArraySize
		if e% 10 == 0:
			torch.save(model.state_dict(), output_dir.joinpath(new_model))
		if e > Args.min_epoch and average_loss_change < 10 and Args.early_stop:
			print('Early stopping at epoch {}'.format(e))
			break
	torch.save(model.state_dict(), output_dir+new_model)

if __name__ == '__main__':
	args = parser.parse_args()
	atm_tcr(args.dataset, args.output_dir, args.new_model, args.old_model, args.epochs)