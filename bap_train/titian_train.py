import os, sys
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import json
from copy import deepcopy
import argparse

import torch
import torch.nn as nn


cwd = os.getcwd()
root_dir = 'attack'
prefix = cwd[:(cwd.find(root_dir)+len(root_dir))] if cwd.find(root_dir)!=-1	 else cwd+f'/{root_dir}' # in case cwd is below root_dir level

REWARD_DIR = Path(prefix).joinpath('TITAN')
sys.path.append(str(REWARD_DIR))

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
DATASET_DIR = Path(prefix).joinpath('bap_attack/data/tcr_split')
OUTPUT_DIR =  Path(prefix).joinpath('bap_attack')
# NEW_MODEL = 'model_list/atmTCR_retrain_1.ckpt'
# OLD_MODEL = 'model_list/atmTCR_retrain.ckpt'
NEW_MODEL = 'model_list/titan_tcr_retrain'