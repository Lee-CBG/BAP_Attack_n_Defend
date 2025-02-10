import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
import argparse
import logging
from copy import deepcopy

import torch
import torch.nn as nn
# set all models in the same level of directory as bap_attack repo
# not sure if this is already appended in the main script. TODO: should not.
"""Script might start at different level based on the debugger setting.
Fix prefix at 'attack' level to accomendate all plugin reward repos.
"""

AMINO_ACIDS = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
root_dir = 'attack'
prefix = str(FILE_PATH)[:(str(FILE_PATH).find(root_dir)+len(root_dir))] if str(FILE_PATH).find(root_dir)!=-1 else str(FILE_PATH)+f'/{root_dir}' # in case cwd is below root_dir level
REWARD_PATH = Path(prefix).joinpath('ERGO')
sys.path.append(str(Path(FILE_PATH).parent))
from utils.data_utils import local_read_candidateTCR

sys.path.append(str(REWARD_PATH))

REWARD_MODEL = 'model_list/ergo_tcr_retrain.pt'
ATTACK_DATA = 'outputs/2025-01-03/06-16-27/iter_0'

logging.basicConfig(
	level=logging.INFO,
	format='%(levelname)s - %(message)s',
	filemode='w',
	filename = Path(prefix).joinpath(f'bap_attack/logs/ergo_infer.log')
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--attack_data', type=str, default=ATTACK_DATA)
parser.add_argument('-m', '--model_dir', type=str, default=REWARD_MODEL)
parser.add_argument('--device', type=int, default=-1)


class Args:
	# Use bi-lstm model as ERGO base architecture
	model_type: str = 'lstm' # 'lstm' or 'ae'
	batch_size: int = 50
	lstm_dim: int = 500
	emb_dim: int = 10
	dropout: float = 0.1
	enc_dim: int = 100
	max_len: int = 28
	ae_file: str = 'TCR_Autoencoder' + '/tcr_ae_dim_' + str(enc_dim) + '.pt'


def convert_data(tcrs, amino_to_ix):
	for i in range(len(tcrs)):
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
	binding = ['p' if p == 1 else 'n' for p in pair[2]]
	pair = [list(t) for t in zip(*[tcr, epi, binding])]
	result = [(p[0], (p[1],), p[2]) for p in pair]
	return result

def ergo(data_dir, model_dir, device):
	logger.debug(torch.cuda.device_count())
	
	if device == -1:
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
		DEVICE = torch.device('cpu')
	else:
		device = 0
		DEVICE = torch.device(f'cuda:{device}')
		torch.cuda.set_device(DEVICE)
	
	logger.debug(f"Device: {DEVICE}")
	data = Path(prefix).joinpath('bap_attack').joinpath(data_dir).joinpath('attack_ergo.csv')
	pair = local_read_candidateTCR(data)
	pair = [t.tolist() for t in pair]

	# if Args.model_type == 'ae':
	# 	import ae_utils  as model_utils 
	# 	from ERGO_models import AutoencoderLSTMClassifier as classifier
	# 	from ERGO import ae_get_lists_from_pairs as get_lists_from_pairs
		
	# 	pep_atox = {amino: index for index, amino in enumerate(['PAD'] + AMINO_ACIDS)}
	# 	tcr_atox = {amino: index for index, amino in enumerate(AMINO_ACIDS + ['X'])}
	# 	# need to tylar sequence to the same length	
	# 	pair = [list(i) for i in pair]
	# 	for i in range(len(pair)):
	# 		if len(pair[i][0]) >= Args.max_len-1:
	# 			pair[i][0] = pair[i][0][:Args.max_len-1]
	# 	pair = [tuple(i) for i in pair]
	# 	tcrs, peps, signs = get_lists_from_pairs(pair, Args.max_len)
	# 	test_batches = model_utils.get_batches(tcrs, peps, signs, tcr_atox, pep_atox, Args.batch_size, Args.max_len)
	# 	model = classifier(10, DEVICE, 28, 21, 100, 50, str(Path(prefix).joinpath('ERGO').joinpath(Args.ae_file)), False)

	if Args.model_type == 'lstm':
		import lstm_utils as model_utils
		from ERGO_models import DoubleLSTMClassifier as classifier
		from ERGO import lstm_get_lists_from_pairs as get_lists_from_pairs
		
		amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + AMINO_ACIDS)}
		pair_test = process_data(pair, amino_to_ix)
		tcrs, peps, signs = get_lists_from_pairs(pair_test)
		# Checked no shuffle here
		test_batches = model_utils.get_batches(tcrs, peps, signs, Args.batch_size)
		model = classifier(10, 500, 0.1, device)
	else:
		raise ValueError(f"Model type {Args.model_type} not supported")

	model.load_state_dict(torch.load(Path(prefix).joinpath('bap_attack').joinpath(model_dir), map_location=torch.device('cpu')))
	model.to(DEVICE)

	model.eval()
	score_model = deepcopy(model)
	score_model.act = nn.Identity()
	y_list = []
	y_score = []
	for batch in test_batches:
		padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
		padded_tcrs = padded_tcrs.to(DEVICE)
		tcr_lens = tcr_lens.to('cpu')
		padded_peps = padded_peps.to(DEVICE)
		pep_lens = pep_lens.to('cpu')
		probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
		y_list.extend(np.array(batch_signs).astype(int))
		y_score.extend(probs.cpu().data.numpy())
	
	dat1 = pd.read_csv(data)
	dat1['yhat'] = np.array(y_score).squeeze().tolist()
	dat1.to_csv(data, index=False, mode='w')

	yhat_list = np.array(y_score).tolist()
	json_output = json.dumps(yhat_list)
	print(json_output)


if __name__ == '__main__':
	args = parser.parse_args()
	ergo(args.attack_data, args.model_dir, args.device)
