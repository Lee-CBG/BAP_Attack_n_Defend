# TODO: check if the token are the same
import os, sys
from pathlib import Path
from dataclasses import dataclass
from typing import Union
# set all models in the same level of directory as bap_attack repo
# not sure if this is already appended in the main script. TODO: should not.
"""Script might start at different level based on the debugger setting.
Fix prefix at 'attack' level to accomendate all plugin reward repos.
"""


# dummy args
@dataclass
class Args:
    blosum: Union[str, None] = 'data/blosum/BLOSUM45'
    lin_size: int = 1024
    max_len_pep: int = 22
    max_len_tcr: int = 20
    drop_rate: float = 0.25
    heads: int = 5

cwd = os.getcwd()
root_dir = 'attack'
prefix = cwd[:(cwd.find(root_dir)+len(root_dir))]
reward_dir = Path(prefix).joinpath('ATM-TCR')
sys.path.append(str(reward_dir))

REWARD_MODEL = 'models/original.ckpt'
# This reward model is only used for the reward generation. 
#For training purpose, go to the plugin reward repo and update weights there.
import torch

from data_loader import define_dataloader, load_embedding, load_data_split
from utils import str2bool, timeSince, get_performance_batchiter, print_performance, write_blackbox_output_batchiter
from attention import Net


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
embedding_matrix = load_embedding(filename=str(reward_dir.joinpath(Args.blosum)))
model = Net(embedding_matrix, Args).to(device)
try: 
	model.load_state_dict(torch.load(reward_dir.joinpath(REWARD_MODEL), map_location=device))
except Exception as e:
	print(e)
	print("Model not loaded. Please check the model definition.")
	sys.exit(1)

model.eval()

