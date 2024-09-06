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
# set all models in the same level of directory as bap_attack repo
# not sure if this is already appended in the main script. TODO: should not.
"""Script might start at different level based on the debugger setting.
Fix prefix at 'attack' level to accomendate all plugin reward repos.
"""


REWARD_MODEL = 'models/ergo_epitope_lstm_model.pt'
ATTACK_DATA = 'log/tmp_epis_tcrs_ergo.csv'
# dummy args
class Args:
    model_type: str = 'lstm' # 'lstm' or 'ae'
    batch_size: int = 50
    lstm_dim: int = 500
    emb_dim: int = 10
    dropout: float = 0.1
    enc_dim: int = 100
    max_len: int = 28
    ae_file: str = 'TCR_Autoencoder' + '/tcr_ae_dim_' + str(enc_dim) + '.pt'

def local_read_candidateTCR(filename):
    pair = []
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
					# dummy labels
                    pair.append((data[1], (data[0]), 'p'))

    infile.close()
    return pair


cwd = os.getcwd()
root_dir = 'attack'
prefix = cwd[:(cwd.find(root_dir)+len(root_dir))] if cwd.find(root_dir)!=-1 else cwd+f'/{root_dir}' # in case cwd is below root_dir level
reward_dir = Path(prefix).joinpath('ERGO')

sys.path.append(str(reward_dir))
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# load generative model acctacks

data_file = str(Path(prefix).joinpath(f'bap_attack/{ATTACK_DATA}'))
pair = local_read_candidateTCR(data_file)

amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
if Args.model_type == 'ae':
    import ae_utils  as model_utils 
    from ERGO_models import AutoencoderLSTMClassifier as classifier
    from ERGO import ae_get_lists_from_pairs as get_lists_from_pairs
    pep_atox = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}
    tcrs, peps, signs = get_lists_from_pairs(pair, Args.max_len)
    test_batches = model_utils.get_full_batches(tcrs, peps, signs, tcr_atox, pep_atox, Args.batch_size, Args.max_len)
    model = classifier(10, device, 28, 21, 100, 50, str(Path(prefix).joinpath('ERGO').joinpath(Args.ae_file)), False)

elif Args.model_type == 'lstm':
    import lstm_utils as model_utils
    from ERGO_models import DoubleLSTMClassifier as classifier
    amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    from ERGO import lstm_get_lists_from_pairs as get_lists_from_pairs
    tcrs, peps, signs = get_lists_from_pairs(pair)
    model_utils.convert_data(tcrs, peps, amino_to_ix)
    test_batches = model_utils.get_full_batches(tcrs, peps, signs, Args.batch_size, amino_to_ix)
    model = classifier(10, 500, 0.1, device)

else:
	raise ValueError(f"Model type {Args.model_type} not supported")

try: 
	model.load_state_dict(torch.load(reward_dir.joinpath(REWARD_MODEL), map_location=device)['model_state_dict'])
except Exception as e:
	print(e)
	print(f"Model not loaded. File at {str(reward_dir.joinpath(REWARD_MODEL))} does not exsists.")
	sys.exit(1)

model.eval()
model.to(device)
score_model = deepcopy(model)
score_model.act = nn.Identity()

y_score = model_utils.predict(score_model, test_batches, device)

dat1 = pd.read_csv(data_file)
data_file_output = str(Path(data_file).parent.joinpath(
     Path(ATTACK_DATA).stem +  f'{Path(REWARD_MODEL).stem}_output' + Path(ATTACK_DATA).suffix))
dat1['yhat'] = np.array(y_score)[:,np.newaxis].tolist()
if not os.path.exists(data_file_output):
	dat1.to_csv(data_file_output, index=False, mode='w')
else:
    dat1.to_csv(data_file_output, header= False, index=False, mode='a')
yhat_list = np.array(y_score)[:,np.newaxis]
yhat_list.reshape(-1)
yhat_list = yhat_list.tolist()
json_output = json.dumps(yhat_list)
print(json_output)
