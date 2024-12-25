import os, sys
from pathlib import Path
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from copy import deepcopy
import argparse
# set all models in the same level of directory as bap_attack repo
# not sure if this is already appended in the main script. TODO: should not.
"""Script might start at different level based on the debugger setting.
Fix prefix at 'attack' level to accomendate all plugin reward repos.
"""


# TODO: Try to rule out dependency in the future version. Need reward model dependency now.
cwd = os.getcwd()
root_dir = 'attack'
prefix = cwd[:(cwd.find(root_dir)+len(root_dir))] if cwd.find(root_dir)!=-1 else cwd+f'/{root_dir}' # in case cwd is below root_dir level
REWARD_DIR = Path(prefix).joinpath('ATM-TCR')
sys.path.append(str(REWARD_DIR))

ATTACK_DATA = 'experiments/result'
REWARD_MODEL ='model_list/atmTCR_tcr_retrain.ckpt'
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


from data_loader import define_dataloader, load_embedding
from attention import Net


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--attack_data', type=str, default=ATTACK_DATA)
parser.add_argument('-m', '--model_dir', type=str, default=REWARD_MODEL)


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

def local_read_candidateTCR(filename):
    peptides=[]
    tcrs=[]
    bound=[]
    # 2 columns: peptides, tcrs, Optional[bounds] candidate pairs
    infile=open(filename, "r")
    for l in infile:
        if l[0] != "#":
            data = l.strip().split("\t")
            if len(data) < 2:
                data = l.strip().split(",")
            if len(data) < 2:
                sys.stderr.write("Problem with input file format!\n")
                sys.stderr.write(l)
                sys.exit(2)
            else:
                if data[0].lower() not in ["epitopes", "peptides"]:
                    peptides.append(data[0])
                    # TODO: check if TCR is longer than 20 amino acids.
                    tmp = data[1]
                    # if len(tmp) > 20:
                    #     # get from left and right 10 amino acids
                    #     tmp = tmp[:10] + tmp[-10:]
                    # tcrs.append(tmp)
                    tcrs.append(data[1])
                    if len(data) > 2:
                        bound.append(data[2])
    peptides_np = np.array(peptides)
    tcrs_np = np.array(tcrs)
    bound_np = np.array(bound) if bound else None
    infile.close()
    return peptides_np, tcrs_np, bound_np


def atm_tcr(data_dir, model_dir):
	data = Path(prefix).joinpath('bap_attack').joinpath(data_dir).joinpath('attack_atmTCR.csv')
	embedding_matrix = load_embedding(filename=Args.blosum)
	model = Net(embedding_matrix, Args).to(DEVICE)
	try: 
		model.load_state_dict(torch.load(Path(prefix).joinpath('bap_attack').joinpath(model_dir)))
	except Exception as e:
		print(e)
		print(f"Model not loaded. File at {Path(prefix).joinpath('bap_attack').joinpath(model_dir)} does not exsists.")
		sys.exit(1)

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
                                    shuffle=Args.shuffle,
	                                batch_size=Args.batch_size,
	                                device=DEVICE)

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
	yhat = np.array(y_score).squeeze()
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
	atm_tcr(args.attack_data, args.model_dir)
