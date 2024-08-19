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
args = parser.parse_args()

model_dir = Path('/mnt/disk07/user/pzhang84/ELMo/ablation_pretraining/pretraining_4_layers_1024')
weights = model_dir/'weights.hdf5'
options = model_dir/'options.json'
embedder  = ElmoEmbedder(options,weights,cuda_device=1) # cuda_device=-1 for CPU
#GeForce RTX 3080 with CUDA capability sm_86 is not compatible with the current PyTorch installation.
#So cuda_device can be -1 or 1 and 2. But cannot be 0.

def ELMo_embeds(x):
    if isinstance(x, float):
        # Handle the case where x is a float
        return [0.0] * 1024  # Return a default value (list of zeros) or take another appropriate action
    else:
        return torch.tensor(embedder.embed_sentence(list(x))).sum(dim=0).mean(dim=0).tolist()

dat1 = pd.read_csv(f'tmp_epis_tcrs.csv')
dat1['tcr_embeds'] = None
dat1['epi_embeds'] = None

for index, row in tqdm(dat1.iterrows(), total=dat1.shape[0]):
    dat1.at[index, 'tcr_embeds'] = ELMo_embeds(row['TCRs'])
    dat1.at[index, 'epi_embeds'] = ELMo_embeds(row['Epitopes'])

dat1.to_pickle(f"tmp_epis_tcrs.pkl")
