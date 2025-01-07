from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
import torch
import pandas as pd
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['pite', 'catelmp-mlp'], default='pite')
parser.add_argument('--data', type=str)
parser.add_argument('--device', type=int, default=-1)
args = parser.parse_args()


#GeForce RTX 3080 with CUDA capability sm_86 is not compatible with the current PyTorch installation.
#So cuda_device can be -1 or 1 and 2. But cannot be 0.
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
root_dir = 'attack'
prefix = str(FILE_PATH)[:(str(FILE_PATH).find(root_dir)+len(root_dir))] if str(FILE_PATH).find(root_dir)!=-1 else str(FILE_PATH)+f'/{root_dir}' # in case cwd is below root_dir level
# pite model
MODEL_DIR = Path(prefix).joinpath('pite/pretraining_4_layers_1024')

weights = MODEL_DIR/'weights.hdf5'
options = MODEL_DIR/'options.json'


def ELMo_embeds(x, embedder):
	if isinstance(x, float):
		# Handle the case where x is a float
		return [0.0] * 1024  # Return a default value (list of zeros) or take another appropriate action
	else:
		if args.model == 'pite':
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

def embedding(model, data, device):
	if device == -1:
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
		DEVICE = torch.device('cpu')
	else:
		device = 0
		DEVICE = torch.device(f'cuda:{device}')
		torch.cuda.set_device(DEVICE)
	
	embedder  = ElmoEmbedder(options,weights,cuda_device=device)
	data_path = Path(prefix).joinpath(f'bap_attack/{data}')
	dat1 = pd.read_csv(data_path)
	dat1['tcr_embeds'] = None
	dat1['epi_embeds'] = None
	# for index, row in tqdm(dat1.iterrows(), total=dat1.shape[0]):
	dat1['tcr_embeds'] = dat1['TCRs'].apply(lambda x: ELMo_embeds(x, embedder))
	dat1['epi_embeds'] = dat1['Epitopes'].apply(lambda x: ELMo_embeds(x, embedder))

	# pad trainValid
	if model == 'pite':
		for i in tqdm(range(len(dat1))):
			dat1.tcr_embeds[i] = padding_helper(dat1.tcr_embeds[i])
			dat1.epi_embeds[i] = padding_helper(dat1.epi_embeds[i])

	dat1.to_pickle(data_path.parent.joinpath(data_path.stem + '.pkl'))

if __name__ == '__main__':
	args = parser.parse_args()
	embedding(args.model, args.data, args.device)