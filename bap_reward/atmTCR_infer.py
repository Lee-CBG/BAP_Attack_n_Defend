import os, sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import json
from copy import deepcopy
import argparse
import logging

import torch
import torch.nn as nn
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
from utils.data_utils import local_read_candidateTCR

sys.path.append(str(REWARD_DIR))

ATTACK_DATA = 'outputs/2024-12-26/03-48-48/result'
REWARD_MODEL ='model_list/atmTCR_tcr_retrain.ckpt'

logging.basicConfig(
	level=logging.INFO,
	format='%(levelname)s - %(message)s',
	filemode='w',
	filename = Path(prefix).joinpath(f'bap_attack/logs/atmTCR_infer.log')
)
logger = logging.getLogger(__name__)


from data_loader import load_embedding, Field_modified, tokenizer, AMINO_MAP, AMINO_MAP_REV
from attention import Net


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--attack_data', type=str, default=ATTACK_DATA)
parser.add_argument('-m', '--model_dir', type=str, default=REWARD_MODEL)
parser.add_argument('--device', type=int, default=-1)


# dummy args
@dataclass
class Args:
	blosum: str = str(REWARD_DIR.joinpath('data/blosum/BLOSUM45'))
	lin_size: int = 1024
	max_len_pep: int = 22
	max_len_tcr: int = 20
	drop_rate: float = 0.25
	heads: int = 5
	padding: str = 'mid'
	batch_size: int = 32
	shuffle: bool = False

def define_dataloader(X_pep, X_tcr, y,
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
	
	# Define vocab
	amino_map = AMINO_MAP
	amino_map_rev = AMINO_MAP_REV
		
	field_pep.build_vocab()
	field_tcr.build_vocab()
	field_y.build_vocab()

	field_pep.vocab.stoi = amino_map
	field_tcr.vocab.stoi = amino_map
	field_pep.vocab.itos = amino_map_rev
	field_tcr.vocab.itos = amino_map_rev
		
	# Define dataloader
	fields = [('X_pep',field_pep), ('X_tcr',field_tcr), ('y',field_y)]
	example = [Example.fromlist([x1,x2,x3], fields) for x1,x2,x3 in zip(X_pep,X_tcr,y)]

	dataset = Dataset(example, fields)
	loader = Iterator(dataset, batch_size=batch_size, device=device, repeat=False, shuffle=False)

	data_loader = dict()
	data_loader['pep_amino_idx'] = field_pep.vocab.itos
	data_loader['tcr_amino_idx'] = field_tcr.vocab.itos
	data_loader['tensor_type'] = torch.cuda.LongTensor if device >=0  else torch.LongTensor
	data_loader['pep_length'] = maxlen_pep
	data_loader['tcr_length'] = maxlen_tcr 
	data_loader['loader'] = loader
	return data_loader


def atm_tcr(data_dir, model_dir, device):
	logger.info('Start ATM-TCR inference')
	if device == -1:
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
		DEVICE = torch.device('cpu')
	else:
		device = 0
		DEVICE = torch.device(f'cuda:{device}')
		torch.cuda.set_device(DEVICE)
		logger.info('device set')

	data = Path(prefix).joinpath('bap_attack').joinpath(data_dir).joinpath('attack_atmTCR.csv')
	embedding_matrix = load_embedding(filename=Args.blosum)
	model = Net(embedding_matrix, Args)
	model.load_state_dict(torch.load(Path(prefix).joinpath('bap_attack').joinpath(model_dir), map_location=torch.device('cpu')))
	logger.info('Model loaded')
	model.to(DEVICE)

	model.eval()
	score_model = deepcopy(model)
	score_model.net[9] = nn.Identity()
	# load generative model acctacks

	x_pep, x_tcr, bound = local_read_candidateTCR(data)
	bound = np.zeros(len(x_pep))
	# TODO: check if generated TCR longer than 20 amino acids.
	
	data_loader = define_dataloader(x_pep, x_tcr, bound, 
									maxlen_pep=Args.max_len_pep,
									maxlen_tcr=Args.max_len_tcr,
									padding=Args.padding,
									batch_size=Args.batch_size,
									device=device)

	logger.info('Data loaded')
	y_score = []
	for batch in data_loader['loader']:
		X_pep, X_tcr, _ = batch.X_pep.to(DEVICE), batch.X_tcr.to(DEVICE), batch.y.to(DEVICE)
		with torch.no_grad():
			# pred = model(X_pep, X_tcr)
			score = score_model(X_pep, X_tcr)
		# y_pred.extend(pred.to('cpu').numpy().tolist())
		y_score.extend(score.to('cpu').numpy().tolist())
	dat1 = pd.read_csv(data)
	# data_file_output = str(data.parent.joinpath(f'{data.stem}_output' + data.suffix))
	yhat = np.round(np.array(y_score).squeeze(),5)
	dat1['yhat'] = yhat
	# if not os.path.exists(data_file_output):
	# 	dat1.to_csv(data_file_output, index=False, mode='w')
	# else:
	# 	dat1.to_csv(data_file_output, header= False, index=False, mode='a')
	dat1.to_csv(data, index=False, mode='w')
	yhat_list = np.array(y_score)
	yhat_list = yhat_list.tolist()
	json_output = json.dumps(yhat_list)
	print(json_output)


if __name__ == '__main__':
	args = parser.parse_args()
	atm_tcr(args.attack_data, args.model_dir, args.device)
