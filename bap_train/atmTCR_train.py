import os, sys
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from collections import deque
import logging
from tqdm import tqdm
import wandb
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.legacy.data import Iterator, Example, Field, Dataset
# set all models in the same level of directory as bap_attack repo
# not sure if this is already appended in the main script. TODO: should not.
"""Script might start at different level based on the debugger setting.
Fix prefix at 'attack' level to accomendate all plugin reward repos.
"""


# TODO: Try to rule out dependency in the future version. Need reward model dependency now.
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
root_dir = 'attack'
prefix = str(FILE_PATH)[:(str(FILE_PATH).find(root_dir)+len(root_dir))] if str(FILE_PATH).find(root_dir)!=-1 else str(FILE_PATH)+f'/{root_dir}' # in case cwd is below root_dir level
REWARD_DIR = Path(prefix).joinpath('ATM-TCR')

sys.path.append(str(Path(FILE_PATH).parent))
from utils.data_utils import load_dataset, shuffle_data
from evaluator.metrics import GroupMetricsLogger


DATASET_DIR = Path(prefix).joinpath('bap_attack/outputs/2025-01-03/04-05-30/iter_0')
OUTPUT_DIR = Path(prefix).joinpath('bap_attack')
NEW_MODEL = 'model_list/atmTCR_tcr_retrain_1.ckpt'
OLD_MODEL = 'model_list/atmTCR_tcr_retrain.ckpt'

sys.path.append(str(REWARD_DIR))
from data_loader import load_embedding, Field_modified, tokenizer, AMINO_MAP, AMINO_MAP_REV
from attention import Net

GROUP2IDX = {'pos': 1, 'neg_healthy': 0, 'neg_shuffle': 2, 'neg_control': 3}
IDX2GROUP = {1: 'pos', 0: 'neg_healthy', 2: 'neg_shuffle', 3: 'neg_control'}


logging.basicConfig(
	level=logging.INFO,
	format='%(levelname)s - %(message)s',
	filemode='w',
	filename = Path(prefix).joinpath(f'bap_attack/logs/atmTCR_train.log')
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=Path, default=DATASET_DIR)
parser.add_argument('-o', '--output_dir', type=Path, default=OUTPUT_DIR)
parser.add_argument('--new_model', type=str, default=NEW_MODEL)
parser.add_argument('--old_model', type=str, default=OLD_MODEL)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--device', type=int, default=0)


@dataclass
class Args:
	blosum: str = str(REWARD_DIR.joinpath('data/blosum/BLOSUM45'))
	lin_size: int = 1024
	max_len_pep: int = 22
	max_len_tcr: int = 20
	drop_rate: float = 0.25
	heads: int = 5
	padding: str = 'mid'
	batch_size: int = 64
	lr: float = 0.001
	early_stop: bool = False
	min_epoch: int = 60
	shuffle = True


def define_dataloader(X_pep, X_tcr, y, groups,
					maxlen_pep=None, maxlen_tcr=None, 
					padding='mid',
					batch_size=50, device=-1):
	
	if maxlen_pep is None: maxlen_pep=max([len(x) for x in X_pep])
	if maxlen_tcr is None: maxlen_tcr=max([len(x) for x in X_tcr])

	# Define Field
	field_pep = Field_modified(tokenize=tokenizer, batch_first=True, 
							pad_type=padding, fix_length=maxlen_pep)					   
	field_tcr = Field_modified(tokenize=tokenizer, batch_first=True, 
							pad_type=padding, fix_length=maxlen_tcr)
	field_y = Field(sequential=False, use_vocab=False, dtype=torch.float32)
	
	field_groups = Field(sequential=False, use_vocab=False, dtype=torch.int8)
	groups = [GROUP2IDX[g] for g in groups]
	# Define vocab
	amino_map = AMINO_MAP
	amino_map_rev = AMINO_MAP_REV
		
	field_pep.build_vocab()
	field_tcr.build_vocab()
	field_y.build_vocab()
	field_groups.build_vocab()

	field_pep.vocab.stoi = amino_map
	field_tcr.vocab.stoi = amino_map
	field_pep.vocab.itos = amino_map_rev
	field_tcr.vocab.itos = amino_map_rev
		
	# Define dataloader
	fields = [('X_pep',field_pep), ('X_tcr',field_tcr), ('y',field_y),  ('groups', field_groups)]
	example = [Example.fromlist([x1,x2,x3,x4], fields) for x1,x2,x3,x4 in zip(X_pep,X_tcr,y, groups)]

	dataset = Dataset(example, fields)
	loader = Iterator(dataset, batch_size=batch_size, device=device, repeat=False, shuffle=True)

	data_loader = dict()
	data_loader['pep_amino_idx'] = field_pep.vocab.itos
	data_loader['tcr_amino_idx'] = field_tcr.vocab.itos
	data_loader['tensor_type'] = torch.cuda.LongTensor if device >=0  else torch.LongTensor
	data_loader['pep_length'] = maxlen_pep
	data_loader['tcr_length'] = maxlen_tcr 
	data_loader['loader'] = loader

	return data_loader

def get_performance_batchiter(loader, model, device='cpu'):
	'''
	print classification performance for binary task

	Args:
	 loader  - data loader
	 model   - classification model
	 loss_ft - loss function
	'''
	model.eval()

	loss = 0
	score, label = [], []
	for batch in loader:

		X_pep, X_tcr, y = batch.X_pep.to(
			device), batch.X_tcr.to(device), batch.y.to(device)
		yhat = model(X_pep, X_tcr)
		y = y.unsqueeze(-1).expand_as(yhat)
		loss += F.binary_cross_entropy(yhat, y, reduction='sum').item()
		score.extend(yhat.data.cpu().tolist())
		label.extend(y.data.cpu().tolist())

	perf = get_performance(score, label)
	perf['loss'] = round(loss, 4)

	return perf


def get_performance(score, label):
	'''
	get classification performance for binary task

	Args:
	 score - 1D np.array or list
	 label - 1D np.array or list
	'''

	accuracy = None
	precision1, precision0 = None, None
	recall1, recall0 = None, None
	f1macro, f1micro = None, None
	auc = None

	# if type(score) is list():
	#	score = np.array(score)
	# if type(label) is list():
	#	label = np.array(label)

	label_pred = [round(s[0]) for s in score]
	accuracy = accuracy_score(label, label_pred)
	precision1 = precision_score(
		label, label_pred, pos_label=1, zero_division=0)
	precision0 = precision_score(
		label, label_pred, pos_label=0, zero_division=0)
	recall1 = recall_score(label, label_pred, pos_label=1, zero_division=0)
	recall0 = recall_score(label, label_pred, pos_label=0, zero_division=0)
	f1macro = f1_score(label, label_pred, average='macro')
	f1micro = f1_score(label, label_pred, average='micro')
	auc = roc_auc_score(np.array(label), np.array(score)) if len(
		np.unique(np.array(label))) != 1 else -1

	ndigits = 4
	performance = {'accuracy': round(accuracy, ndigits),
					'precision1': round(precision1, ndigits), 'precision0': round(precision0, ndigits),
					'recall1': round(recall1, ndigits), 'recall0': round(recall0, ndigits),
					'f1macro': round(f1macro, ndigits), 'f1micro': round(f1micro, ndigits),
					'auc': round(auc, ndigits)}
	tn, fp, fn, tp = confusion_matrix(label, label_pred, labels=[0, 1]).ravel()
	print(tn, fp, fn, tp)
	return performance


def train(model, train_loader, optimizer, device):
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


def atm_tcr(dataset_dir, output_dir, new_model, old_model, epochs, device):
	if device == -1:
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
		DEVICE = torch.device('cpu')
	else:
		device = 0
		DEVICE = torch.device(f'cuda:{device}')
		torch.cuda.set_device(DEVICE)

	if not old_model:
		old_model = None
	else:
		Args.lr *= 0.01
	logger.info('==========================================================')
	embedding_matrix = load_embedding(filename=Args.blosum)
	model = Net(embedding_matrix, Args).to(DEVICE)
	if old_model is None:
		train_epi, train_tcr, train_label, train_group = load_dataset(dataset_dir, 'training.pkl', shuffle=True, partial=1)
		val_epi, val_tcr, val_label, val_group = load_dataset(dataset_dir, 'training.pkl', shuffle=False, partial=-0.2)
	else:
		model.load_state_dict(torch.load(output_dir.joinpath(old_model), map_location=DEVICE))
		logger.info(f'Model loaded  @{output_dir.joinpath(old_model)}')
		train_epi, train_tcr, train_label, train_group = load_dataset(dataset_dir, 'trainData.pkl', shuffle=True, partial=0.8)
		val_epi, val_tcr, val_label, val_group = load_dataset(dataset_dir, 'trainData.pkl', shuffle=False, partial=-0.2)
	logger.info(f'Loading training dataset  @{dataset_dir}')
	
	# Conformed that define_dataloader has no shuffle operation
	train_loader = define_dataloader(train_epi, train_tcr, train_label, train_group,
									 Args.max_len_pep, Args.max_len_tcr,
									 padding=Args.padding,
									 batch_size=Args.batch_size, device=device)
	val_loader = define_dataloader(val_epi, val_tcr, val_label, val_group,
									Args.max_len_pep, Args.max_len_tcr,									padding=Args.padding,
									batch_size=Args.batch_size, device=device)
	
	optimizer = optim.Adam(model.parameters(), lr=Args.lr)
	lossArraySize = 10
	lossArray = deque([sys.maxsize], maxlen=lossArraySize)
	auc = []
	accuracy = []
	if old_model is not None:
		group_logger = GroupMetricsLogger(groups=['pos', 'neg_healthy', 'neg_shuffle', 'neg_control'], device=DEVICE)
		train_list = [(t[0].X_pep, t[0].X_tcr, t[0].y, t[0].groups) for t in zip(train_loader['loader'])]
		X1_train_split, X2_train_split, y_train_split, group_train_split = zip(*train_list)
		X1_train_split = torch.cat(X1_train_split)
		X2_train_split = torch.cat(X2_train_split)
		y_train_split = torch.cat(y_train_split)
		group_train_split = torch.cat(group_train_split).tolist()
		group_train_split = [IDX2GROUP[g] for g in group_train_split]
		group_train_split = np.array(group_train_split)
		
		val_list =  [(t[0].X_pep, t[0].X_tcr, t[0].y, t[0].groups) for t in zip(val_loader['loader'])]
		X1_val, X2_val, y_val, group_val = zip(*val_list)
		X1_val = torch.cat(X1_val)
		X2_val = torch.cat(X2_val)
		y_val = torch.cat(y_val)
		group_val = torch.cat(group_val).tolist()
		group_val = [IDX2GROUP[g] for g in group_val]
		group_val = np.array(group_val)
		wandb.init(project='bap_defendings_atmTCR', name=f'{dataset_dir.relative_to(dataset_dir.parents[2])}')

	for epoch in tqdm(range(epochs)):
		train(model, train_loader['loader'], optimizer, DEVICE)
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
		if old_model is not None:
				group_logger.on_epoch_end(
			model=model,
			train_data=(X1_train_split, X2_train_split),
			train_labels=y_train_split,
			train_groups=group_train_split,
			val_data=(X1_val, X2_val),
			val_labels=y_val,
			val_groups=group_val,
			epoch=epoch
		)

		# train_epi, train_tcr, train_label = shuffle_data(train_epi, train_tcr, train_label)

	torch.save(model.state_dict(), output_dir.joinpath(new_model))
	
	if old_model is None:
		test_epi, test_tcr, test_label, test_group = load_dataset(dataset_dir, 'testing.pkl', shuffle=False, partial=1)
	else:
		test_epi, test_tcr, test_label, test_group = load_dataset(dataset_dir, 'testData.pkl', shuffle=False, partial=1)
	test_loader = define_dataloader(test_epi, test_tcr, test_label, test_group,
								maxlen_pep=Args.max_len_pep,
								maxlen_tcr=Args.max_len_tcr,
								padding=Args.padding,
								batch_size=Args.batch_size,
								device=device)
	logger.info(f'Loading testing dataset from {dataset_dir}')
	yhat_list = []
	y_list = []
	model.load_state_dict(torch.load(output_dir.joinpath(new_model), map_location=DEVICE))
	model = model.eval()
	for batch in test_loader['loader']:
		X_pep, X_tcr, y = batch.X_pep.to(DEVICE), batch.X_tcr.to(DEVICE), batch.y.to(DEVICE)
		yhat_list.append(model(X_pep, X_tcr).cpu().detach().numpy())
		y_list.append(y.cpu().detach().numpy())
	yhat = np.concatenate(yhat_list)
	y = np.concatenate(y_list)
	roc = roc_auc_score(y, yhat)
	logger.info(f'AUC: {roc}')
	acc = accuracy_score(y, np.round(yhat))
	logger.info(f'ACC: {acc}')
	wandb.log({'AUC': roc, 'ACC': acc})
	logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


if __name__ == '__main__':
	args = parser.parse_args()
	atm_tcr(args.dataset, args.output_dir, args.new_model, args.old_model, args.epochs, args.device)