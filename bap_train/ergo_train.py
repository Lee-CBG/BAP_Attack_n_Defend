import os, sys
from pathlib import Path
import numpy as np
import argparse
from random import shuffle
import logging
from tqdm import tqdm
import wandb
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, log_loss

import torch
import torch.nn as nn
import torch.optim as optim

AMINO_ACIDS = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
root_dir = 'attack'
prefix = str(FILE_PATH)[:(str(FILE_PATH).find(root_dir)+len(root_dir))] if str(FILE_PATH).find(root_dir)!=-1 else str(FILE_PATH)+f'/{root_dir}' # in case cwd is below root_dir level
REWARD_PATH = Path(prefix).joinpath('ERGO')
sys.path.append(str(Path(FILE_PATH).parent))
from utils.data_utils import load_dataset

sys.path.append(str(REWARD_PATH))

DATASET_DIR = Path(prefix).joinpath('bap_attack/outputs/2025-01-03/16-59-38/iter_0')
OUTPUT_DIR = Path(prefix).joinpath('bap_attack')
NEW_MODEL = 'model_list/ergo_tcr_retrain_1.pt'
OLD_MODEL = 'model_list/ergo_tcr_retrain.pt'

logging.basicConfig(
	level=logging.INFO,
	format='%(levelname)s - %(message)s',
	filemode='w',
	filename = Path(prefix).joinpath(f'bap_attack/logs/ergo_train.log')
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=Path, default=DATASET_DIR)
parser.add_argument('-o', '--output_dir', type=Path, default=OUTPUT_DIR)
parser.add_argument('--new_model', type=str, default=NEW_MODEL)
parser.add_argument('--old_model', type=str, default=OLD_MODEL)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--device', type=int, default=1)


class Args:
	# Use bi-lstm model as ERGO base architecture
	model_type: str = 'lstm' # 'lstm' or 'ae'
	batch_size: int = 256
	lstm_dim: int = 500
	emb_dim: int = 10
	dropout: float = 0.1
	enc_dim: int = 100
	max_len: int = 21
	weight_decay: float = 0.00
	ae_file: str = 'TCR_Autoencoder' + '/tcr_ae_dim_' + str(enc_dim) + '.pt'
	lr: float = 1e-3
	

class GroupMetricsLogger:
	def __init__(self, groups, device='cpu'):
		"""
		groups (list): List of group names, e.g., ['pos', 'neg_healthy', 'neg_shuffle'].
		"""
		self.groups = groups
		self.device = device

	def compute_group_metrics(self, model, features, labels, groups, split_name, epoch):
		"""
		Compute and log metrics for each group.
		Args:
			model (torch.nn.Module): Trained PyTorch model.
			features (tuple): Input features (X1, X2).
			labels (torch.Tensor): True labels.
			groups (np.ndarray): Group identifiers for each sample.
			split_name (str): Name of the data split (e.g., "Training" or "Validation").
			epoch (int): Current epoch number.
		"""
		model.eval()
		grps = []
		prdcs= []
		lbs = []
		# pipe in batches
		for i in range(len(labels)):
			with torch.no_grad():
				X1, X1_len, X2, X2_len = [features[j][i] for j in range(len(features))]
				X1 = X1.to(self.device)
				X1_len = X1_len.to('cpu')
				X2 = X2.to(self.device)
				X2_len = X2_len.to('cpu')
				prdcs.extend(model(X1, X1_len, X2, X2_len).cpu().numpy())
				grps.extend(groups[i])
				lbs.extend(labels[i])
		prdcs = np.array(prdcs)
		lbs = np.array(lbs)
		grps = np.array(grps)

		for group in self.groups:
				# Mask for the current group
			group_mask = (grps == group)
			group_true = lbs[group_mask]
			group_pred = prdcs[group_mask].squeeze()
				# Compute metrics
			if len(group_true) > 0:
				group_loss = log_loss(group_true, group_pred, labels=[0, 1])
				group_acc = accuracy_score(group_true, np.round(group_pred))
			else:
				group_loss = None
				group_acc = None
			# Log metrics to wandb
			wandb.log({
				f"{split_name}/{group}_loss": group_loss if group_loss is not None else np.nan,
				f"{split_name}/{group}_accuracy": group_acc if group_acc is not None else np.nan,
				f"{split_name}/{group}_count": np.sum(group_mask),
				"epoch": epoch
			})

	def on_epoch_end(self, model, train_data, train_labels, train_groups,
					 val_data, val_labels, val_groups, epoch):

		self.compute_group_metrics(model, train_data, train_labels, train_groups, "Training", epoch)
		self.compute_group_metrics(model, val_data, val_labels, val_groups, "Validation", epoch)
		
def convert_data(tcrs, amino_to_ix):
	for i in range(len(tcrs)):
		if any(letter.islower() for letter in tcrs[i]):
			logger.debug(f'Lower case letter found in {tcrs[i]}')
		try: 
			tcrs[i] = [amino_to_ix[amino.upper()] for amino in tcrs[i]]
		except KeyError:
			# guess O is miss match of Q
			tcrs[i] = np.char.replace(tcrs[i], 'O', 'K')
			tcrs[i] = np.char.replace(tcrs[i], 'B', 'D')
			tcrs[i] = np.char.replace(tcrs[i], 'U', 'C')
			tcrs[i] = np.char.replace(tcrs[i], 'Z', 'E')
			tcrs[i] = tcrs[i].tolist()
			tcrs[i] = [amino_to_ix[amino.upper()] for amino in tcrs[i]]
	return tcrs

def process_data(pair, amino_to_ix):
	# need to convert data to adapt to the generative model
	tcr = convert_data(pair[1], amino_to_ix)
	epi = convert_data(pair[0], amino_to_ix)
	epi = [t[0] for t in epi]
	# binding = ['p' if p == 1 else 'n' for p in pair[2]]
	binding = pair[2]
	groups = pair[3]
	pair = [list(t) for t in zip(*[tcr, epi, binding, groups])]
	result = [(p[0], (p[1],), p[2], p[3]) for p in pair]
	return result

def group_batches(groups, batch_size):
	# split group in to batches of size batch_size
	batches = []
	for i in range(0, len(groups), batch_size):
		batches.append(np.array(groups[i:i+batch_size]))
	return batches


def train(batches, model, loss_function, optimizer, device):
	model.train()
	shuffle(batches)
	total_loss = 0
	for batch in batches:
		padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
		padded_tcrs = padded_tcrs.to(device)
		tcr_lens = tcr_lens.to('cpu')
		padded_peps = padded_peps.to(device)
		pep_lens = pep_lens.to('cpu')
		batch_signs = torch.tensor(np.array(batch_signs)[:,np.newaxis], dtype=torch.float).to(device)
		model.zero_grad()
		probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)

		weights = batch_signs * 0.84 + (1-batch_signs) * 0.14
		loss_function.weight = weights.float()
		loss = loss_function(probs, batch_signs)
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
	return total_loss / len(batches)

def evaluate(model, batches, device):
	model.eval()
	true = []
	scores = []
	shuffle(batches)
	for batch in batches:
		padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
		# Move to GPU
		padded_tcrs = padded_tcrs.to(device)
		tcr_lens = tcr_lens.to('cpu')
		padded_peps = padded_peps.to(device)
		pep_lens = pep_lens.to('cpu')
		probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
		# print(np.array(batch_signs).astype(int))
		# print(probs.cpu().data.numpy())
		true.extend(np.array(batch_signs).astype(int))
		scores.extend(probs.cpu().data.numpy())
	# Return auc score
	auc = roc_auc_score(true, scores)
	fpr, tpr, thresholds = roc_curve(true, scores)
	return auc, (fpr, tpr, thresholds)


def ergo(dataset_dir, output_dir, new_model, old_model, epochs, device):
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
	logger.info('============================================================')
	logger.info(f'Learing rate: {Args.lr}')
	logger.info(f'Loading dataset from {dataset_dir}')
	if old_model is None:
		pair_train = load_dataset(dataset_dir, 'training.pkl', shuffle=True, partial=0.8)
		pair_val = load_dataset(dataset_dir, 'training.pkl', shuffle=False, partial=-0.2)
		pair_test = load_dataset(dataset_dir, 'testing.pkl', shuffle=False, partial=1)
	else:
		pair_train = load_dataset(dataset_dir, 'trainData.pkl', shuffle=True, partial=0.8)
		pair_val = load_dataset(dataset_dir, 'trainData.pkl', shuffle=False, partial=-0.2)
		pair_test = load_dataset(dataset_dir, 'testData.pkl', shuffle=False, partial=1)

	# if Args.model_type == 'ae':
	# 	import ae_utils  as model_utils 
	# 	from ERGO_models import AutoencoderLSTMClassifier as classifier
	# 	from ERGO import ae_get_lists_from_pairs as get_lists_from_pairs

	# 	# TODO: test later
	# 	pep_atox = {amino: index for index, amino in enumerate(['PAD'] + AMINO_ACIDS)}
	# 	tcr_atox = {amino: index for index, amino in enumerate(AMINO_ACIDS + ['X'])}
		
	# 	# Checked: no shuffle in the processing operations
	# 	train_tcrs, train_peps, train_signs = get_lists_from_pairs(pair_train[:3], Args.max_len)
	# 	group_train_split = np.array([p for p in pair_train[3]])
	# 	train_batches = model_utils.get_batches(train_tcrs, train_peps, train_signs, tcr_atox, pep_atox, Args.batch_size, Args.max_len)
	# 	# test
	# 	val_tcrs, val_peps, val_signs = get_lists_from_pairs(pair_val[:3], Args.max_len)
	# 	group_val = np.array([p for p in pair_val[3]])
	# 	val_batches = model_utils.get_batches(val_tcrs, val_peps, val_signs, tcr_atox, pep_atox, Args.batch_size, Args.max_len)
	# 	test_tcrs, test_peps, test_signs = get_lists_from_pairs(pair_test[:3], Args.max_len)
	# 	test_batches = model_utils.get_batches(test_tcrs, test_peps, test_signs, tcr_atox, pep_atox, Args.batch_size, Args.max_len)	
	# 	# need to tylar sequence to the same length	
	# 	model = classifier(10, DEVICE, 28, 21, 100, 50, str(Path(prefix).joinpath('ERGO').joinpath(Args.ae_file)), False)

	if Args.model_type == 'lstm':
		import lstm_utils as model_utils
		from ERGO_models import DoubleLSTMClassifier as classifier
		
		amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + AMINO_ACIDS)}			
		pair_train = [t.tolist() for t in pair_train]
		processed_train = process_data(pair_train, amino_to_ix)
		train_tcrs, train_peps, train_signs, train_groups = [t for t in zip(*processed_train)]
		pair_val = [t.tolist() for t in pair_val]
		processed_val = process_data(pair_val, amino_to_ix)
		val_tcrs, val_peps, val_signs, val_groups = [t for t in zip(*processed_val)]
		logger.info(f'Data processed')	
		train_batches = model_utils.get_batches(train_tcrs, train_peps, train_signs, Args.batch_size)
		group_train_split = group_batches(train_groups, Args.batch_size)
		val_batches = model_utils.get_batches(val_tcrs, val_peps, val_signs, Args.batch_size)
		group_val = group_batches(val_groups, Args.batch_size)
		
		pair_test = [t.tolist() for t in pair_test]
		processed_test = process_data(pair_test, amino_to_ix)
		test_tcrs, test_peps, test_signs, _ = [t for t in zip(*processed_test)]
		test_batches = model_utils.get_batches(test_tcrs, test_peps, test_signs, Args.batch_size)
		logger.info(f'Batches created')
		# Train the model
		model = classifier(10, 500, 0.1, DEVICE)
	else:
		raise ValueError('Model type not recognized')
	
	logger.info(f'Model loaded')
	if old_model is not None:
		model.load_state_dict(torch.load(output_dir.joinpath(old_model), map_location=DEVICE))
		logger.info(f'Model loaded  @{output_dir.joinpath(old_model)}')
	else:
		logger.info(f'Training model from scratch')
	
	if old_model is not None:
		group_logger = GroupMetricsLogger(groups=['pos', 'neg_healthy', 'neg_shuffle', 'neg_control'], device=DEVICE)
		X1_train_split, X1_train_len, X2_train_split, X2_train_len, y_train_split  = [list(t) for t in zip(*train_batches)]
				
		X1_val, X1_val_len, X2_val, X2_val_len, y_val = [list(t) for t in zip(*val_batches)]
		
		wandb.init(project='bap_defendings_ergoLSTM', name=f'{dataset_dir.relative_to(dataset_dir.parents[2])}')

	model.to(DEVICE)
	losses = []
	loss_func = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=Args.lr, weight_decay=Args.weight_decay)
	best_auc = 0
	for epoch in tqdm(range(epochs)):
		logger.info(f'Epoch {epoch}')
		loss = train(train_batches, model, loss_func, optimizer, DEVICE)
		losses.append(loss)
		train_auc, _ = evaluate(model, train_batches, device)
		logger.info(f'Train Loss: {loss} - Train AUC: {train_auc}')
		val_auc, _ = evaluate(model, val_batches, device)
		logger.info(f'Validation AUC: {val_auc}')
		if val_auc > best_auc:
			best_auc = val_auc
			torch.save(model.state_dict(), output_dir.joinpath(new_model))
			logger.info(f'Best AUC: {best_auc}')
			logger.info(f'Model saved to {output_dir.joinpath(new_model)}')
		if old_model is not None:
				group_logger.on_epoch_end(
			model=model,
			train_data=(X1_train_split, X1_train_len, X2_train_split, X2_train_len),
			train_labels=y_train_split,
			train_groups=group_train_split,
			val_data=(X1_val, X1_val_len, X2_val, X2_val_len),
			val_labels=y_val,
			val_groups=group_val,
			epoch=epoch
		)	
	
	yhat_list = []
	y_list = []
	model.load_state_dict(torch.load(output_dir.joinpath(new_model), map_location=DEVICE))
	model = model.eval()
	for batch in test_batches:
		padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
		padded_tcrs = padded_tcrs.to(DEVICE)
		tcr_lens = tcr_lens.to('cpu')
		padded_peps = padded_peps.to(DEVICE)
		pep_lens = pep_lens.to('cpu')
		probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
		y_list.extend(np.array(batch_signs).astype(int))
		yhat_list.extend(probs.cpu().data.numpy())
	# TODO: fix later
	roc = roc_auc_score(y_list, yhat_list)
	logger.info(f'AUC: {roc}')
	acc = accuracy_score(np.array(y_list), np.round(np.array(yhat_list)))
	logger.info(f'ACC: {acc}')
	wandb.log({'AUC': roc, 'ACC': acc})
	logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


if __name__ == '__main__':
	args = parser.parse_args()
	ergo(args.dataset, args.output_dir, args.new_model, args.old_model, args.epochs, args.device)