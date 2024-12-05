from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
import torch
import pandas as pd
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--epi', type=str)
parser.add_argument('--n_seq', type=int, default=100)
parser.add_argument('--pad', type=bool, default=True)
parser.add_argument('--folder', type=str, default='data_rlhf')
args = parser.parse_args()


#GeForce RTX 3080 with CUDA capability sm_86 is not compatible with the current PyTorch installation.
#So cuda_device can be -1 or 1 and 2. But cannot be 0.
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
root_dir = 'attack'
prefix = str(FILE_PATH)[:(str(FILE_PATH).find(root_dir)+len(root_dir))] if str(FILE_PATH).find(root_dir)!=-1 else str(FILE_PATH)+f'/{root_dir}' # in case cwd is below root_dir level
MODEL_DIR = Path(prefix).joinpath('pite/pretraining_4_layers_1024')
DEVICE = -1
weights = MODEL_DIR/'weights.hdf5'
options = MODEL_DIR/'options.json'


embedder  = ElmoEmbedder(options,weights,cuda_device=DEVICE) # cuda_device=-1 for CPU


def ELMo_embeds(x):
    if isinstance(x, float):
        # Handle the case where x is a float
        return [0.0] * 1024  # Return a default value (list of zeros) or take another appropriate action
    else:
        if args.pad:
            return torch.tensor(embedder.embed_sentence(list(x))).mean(dim=0).tolist()
        else:
            return torch.tensor(embedder.embed_sentence(list(x))).sum(dim=0).mean(dim=0).tolist()


def padding_helper(seq, padded_len=22):
    if len(seq) >= padded_len:
        seq = seq[:padded_len]
    else:
        suf = [0] * 1024
        while len(seq) < padded_len:
            seq.append(suf)
    return seq


if args.pad:
	dat1 = pd.read_csv(Path(prefix).joinpath(f'bap_attack/{args.folder}/attack_pite.csv'))
else:
	dat1 = pd.read_csv(Path(prefix).joinpath(f'bap_attack/{args.folder}/attack_catelmp-mlp.csv'))
dat1['tcr_embeds'] = None
dat1['epi_embeds'] = None

for index, row in tqdm(dat1.iterrows(), total=dat1.shape[0]):
    dat1.at[index, 'tcr_embeds'] = ELMo_embeds(row['TCRs'])
    dat1.at[index, 'epi_embeds'] = ELMo_embeds(row['Epitopes'])

# pad trainValid
if args.pad:
  for i in tqdm(range(len(dat1))):
    dat1.tcr_embeds[i] = padding_helper(dat1.tcr_embeds[i])
    dat1.epi_embeds[i] = padding_helper(dat1.epi_embeds[i])

if args.pad:
	dat1.to_pickle(Path(prefix).joinpath(f'bap_attack/{args.folder}/attack_pite.pkl'))
else:
     dat1.to_pickle(Path(prefix).joinpath(f'bap_attack/{args.folder}/attack_catelmp-mlp.pkl'))