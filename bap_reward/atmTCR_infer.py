import os, sys
from pathlib import Path
from dataclasses import dataclass
from typing import Union
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

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
REWARD_MODEL = Path(prefix).joinpath('backup_list/atm_tcr-tcr.ckpt')
# ATTACK_DATA = 'log/tmp_epis_tcrs_atm-tcr.csv'
# ATTACK_DATA = Path(prefix).joinpath('metrics/dat_benchmarking.csv')
ATTACK_DATA = Path(prefix).joinpath('experiments/result/attack_atmTCR.csv')


from data_loader import define_dataloader, load_embedding
from attention import Net


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--attack_data', type=Path, default=ATTACK_DATA)
parser.add_argument('-m', '--reward_model', type=Path, default=REWARD_MODEL)


# dummy args
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
                    tcrs.append(data[1])
                    if len(data) > 2:
                        bound.append(data[2])
    peptides_np = np.array(peptides)
    tcrs_np = np.array(tcrs)
    bound_np = np.array(bound) if bound else None
    infile.close()
    return peptides_np, tcrs_np, bound_np


def atm_tcr(data_dir, model_dir):
	embedding_matrix = load_embedding(filename=Args.blosum)
	model = Net(embedding_matrix, Args).to(DEVICE)
	try: 
		model.load_state_dict(torch.load(model_dir), map_location=DEVICE)
	except Exception as e:
		print(e)
		print(f"Model not loaded. File at {model_dir} does not exsists.")
		sys.exit(1)

	model.eval()
	score_model = deepcopy(model)
	score_model.net[9] = nn.Identity()
	# load generative model acctacks
	data_file = str(data_dir)
	x_pep, x_tcr, bound = local_read_candidateTCR(data_file)
	bound = np.zeros(len(x_pep))
	data_loader = define_dataloader(x_pep, x_tcr, bound, 
	                                maxlen_pep=Args.max_len_pep,
	                                maxlen_tcr=Args.max_len_tcr,
	                                padding=Args.padding,
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
	dat1 = pd.read_csv(data_file)
	data_file_output = str(data_file.parent.joinpath(
     data_dir.stem + f'{Path(model_dir).stem}_output' + data_dir.suffix))
	dat1['yhat'] = y_score

	if not os.path.exists(data_file_output):
		dat1.to_csv(data_file_output, index=False, mode='w')
	else:
		dat1.to_csv(data_file_output, header= False, index=False, mode='a')
	yhat_list = np.array(y_score)
	yhat_list.reshape(-1)
	yhat_list = yhat_list.tolist()
	json_output = json.dumps(yhat_list)
	print(json_output)



if __name__ == '__main__':
	args = parser.parse_args()
	atm_tcr(args.attack_data, args.reward_model)
