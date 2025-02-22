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
from copy import deepcopy
# set all models in the same level of directory azs bap_attack repo
# not sure if this is already appended in the main script. TODO: should not.exit()
"""Script might start at different level based on the debugger setting.
Fix prefix at 'attack' level to accomendate all plugin reward repos.
"""

cwd = os.getcwd()
root_dir = 'attack'
prefix = cwd[:(cwd.find(root_dir)+len(root_dir))] if cwd.find(root_dir)!=-1 else cwd+f'/{root_dir}' # in case cwd is below root_dir level
REWARD_DIR = Path(prefix).joinpath('TITAN')
sys.path.append(str(REWARD_DIR))

ATTACK_DATA = 'outputs/2024-12-26/03-48-48/result'
REWARD_MODEL ='model_list/atmTCR_tcr_retrain.pt'
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
SPECIAL_TOKENS = ["<PAD>", "<tcr>", "<eotcr>", "[CLS]", "[BOS]", "[MASK]", "[SEP]", "<epi>", "<eoepi>", "$", '<unk>']

from paccmann_predictor.models import MODEL_FACTORY
from pytoda.datasets import ProteinProteinInteractionDataset
from pytoda.proteins import ProteinFeatureLanguage, ProteinLanguage
from pytoda.smiles.smiles_language import SMILESTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--attack_data', type=str, default=ATTACK_DATA)
parser.add_argument('-m', '--model_dir', type=str, default=REWARD_MODEL)
parser.add_argument('-h', '--device', type=int, default=DEVICE)


with open(REWARD_DIR.joinpath('owndata/owndata').joinpath('model_params.json')) as fp:
	default_params = json.load(fp)

@dataclass
class Params:
	model_type: str = 'bimodal_mca'
	model_path: str = str(REWARD_DIR.joinpath('owndata/owndata')) # this is for pretrained language model
	param: dict = field(default_factory=dict)
	ext: str = 'csv'

Params.param = default_params


def _filter(s):
	if type(s) is not str:
		return False
	if s is None:
		return False
	return not any([c in s for c in SPECIAL_TOKENS])

def process_titan(file, ext):
	# deal with invalid tcr
	contents = pd.read_csv(file)
	validIndex = contents['TCRs'].apply(lambda x: _filter(x))
	validContents = contents[validIndex]
	
	log_dir = Path(file).parent
	tcr_set = validContents['TCRs'].unique()
	epi_set = validContents['Epitopes'].unique()
	tcr_rank = np.arange(2, len(tcr_set)+2)
	epi_rank = np.arange(3, len(epi_set)+3)
	epi_rank = np.arange(3, len(epi_set)+3)
	tcr_pd = pd.DataFrame({"tcr_set": tcr_set, "tcr_rank": tcr_rank})
	epi_pd = pd.DataFrame({"epi_set": epi_set, "epi_rank": epi_rank})
	tcr_pd.to_csv(f'{log_dir}/tcr.csv', index=False, header=False, sep='\t')
	epi_pd.to_csv(f'{log_dir}/epitopes.csv', index=False, header=False, sep='\t')
	tcr_readin = pd.read_csv(f'{log_dir}/tcr.csv', header=None, sep='\t')
	tcr_dict = dict(zip(tcr_readin[0], tcr_readin[1]))
	epi_readin = pd.read_csv(f'{log_dir}/epitopes.csv', header=None, sep='\t')
	epi_dict = dict(zip(epi_readin[0], epi_readin[1]))
	train_dataset_epi_ready = pd.DataFrame({'ligand_name': validContents['Epitopes'].map(epi_dict).fillna(-1),
										'sequence_id': validContents['TCRs'].map(tcr_dict).fillna(-1),
										'label': [1]*len(validContents)})
	train_dataset_epi_ready.to_csv(f'{log_dir}/attack.csv',  index=True, sep=',')
	if ext == 'smi':
		pass
	return contents[validIndex].index




def titan(data_dir, model_dir, device):
	data_full = Path(prefix).joinpath('bap_attack').joinpath(data_dir).joinpath('attack_titan.csv')
	model_full = Path(prefix).joinpath('bap_attack').joinpath(model_dir)

	smiles_language = SMILESTokenizer.from_pretrained(Params.model_path)
	smiles_language.set_encoding_transforms(
		randomize=None,
		add_start_and_stop=Params.param.get('ligand_start_stop_token', True),
		padding=Params.param.get('ligand_padding', True),
		padding_length=Params.param.get('ligand_padding_length', True),
	)
	smiles_language.set_smiles_transforms(
		augment=False,
		canonical=Params.param.get('smiles_canonical', False),
		kekulize=Params.param.get('smiles_kekulize', False),
		all_bonds_explicit=Params.param.get('smiles_bonds_explicit', False),
		all_hs_explicit=Params.param.get('smiles_all_hs_explicit', False),
		remove_bonddir=Params.param.get('smiles_remove_bonddir', False),
		remove_chirality=Params.param.get('smiles_remove_chirality', False),
		selfies=Params.param.get('selfies', False),
		sanitize=Params.param.get('sanitize', False)
	)
	if Params.param.get('receptor_embedding', 'learned') == 'predefined':
		protein_language = ProteinFeatureLanguage.load(
			os.path.join(Params.model_path, 'protein_language.pkl')
		)
	else:
		protein_language = ProteinLanguage.load(
			os.path.join(Params.model_path, 'protein_language.pkl')
		)
	model_fn = Params.param.get('model_fn', Params.model_type)
	model = MODEL_FACTORY[model_fn](Params.param).to(DEVICE)
	model._associate_language(smiles_language)
	model._associate_language(protein_language)

	try:
		# retrained model directory
		model_file = os.path.join(
		model_full, 'weights', 'best_ROC-AUC_bimodal_mca.pt'
		)
		model.load(model_full, map_location=DEVICE)
	except Exception as e:
		print(e)
		print(f"Model not loaded. File at {model_file} does not exsists.")
		sys.exit(1)	
	model.eval()
	score_model = deepcopy(model)
	score_model.final_dense[1] = nn.Identity()

# load generatived attacks and transform them to the model acceptable dataset
	if Params.ext == 'csv':
		# load tcr.csv, epitope.csv formation
		valid_idx = process_titan(data_full, Params.ext)
		log_dir = Path(data_full).parent
		eval_dataset = ProteinProteinInteractionDataset(
			sequence_filepaths=[[f'{log_dir}/epitopes.csv'], [f'{log_dir}/tcr.csv']],
			entity_names=['ligand_name', 'sequence_id'],
			labels_filepath=f'{log_dir}/attack.csv',
			annotations_column_names=['label'],
			protein_languages=protein_language,
			padding_lengths=[
			Params.param.get('ligand_padding_length', None),
			Params.param.get('receptor_padding_length', None)
			],
			paddings=Params.param.get('ligand_padding', True),
			add_start_and_stops=Params.param.get('add_start_stop_token', True),
			augment_by_reverts=Params.param.get('augment_protein', False),
			randomizes=Params.param.get('randomize', False),
			iterate_datasets=True
		)
		eval_loader = torch.utils.data.DataLoader(
			dataset=eval_dataset,
			batch_size=512,
			shuffle=False,
			drop_last=False,
			num_workers=Params.param.get('num_workers', 0),
		)

		result = []
		for ind, (ligand, receptors, y) in enumerate(eval_loader):
			torch.cuda.empty_cache()
			score, pred_dict = score_model(ligand.to(DEVICE), receptors.to(DEVICE))
			result.extend(score.to('cpu').detach().numpy().tolist())
		dat1 = pd.read_csv(data_full)

		dat1['yhat'] = -np.inf
		dat1.loc[valid_idx, 'yhat'] = np.round(np.array(result).squeeze().tolist(),5)
		dat1.to_csv(data_full, index=False, mode='w')
		yhat_list = np.array(result)
		yhat_list = yhat_list.tolist()
		json_output = json.dumps(yhat_list)
		print(json_output)


if __name__ == '__main__':
	args = parser.parse_args()
	titan(args.attack_data, args.model_dir, args.device)