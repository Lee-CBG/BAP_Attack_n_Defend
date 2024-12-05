import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', type=str, default='data_bap')
parser.add_argument('-b', '--bap', type=str, default='pite')


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
root_dir = 'attack'
PREFIX = str(FILE_PATH)[:(str(FILE_PATH).find(root_dir)+len(root_dir))] if str(FILE_PATH).find(root_dir)!=-1 else str(FILE_PATH)+f'/{root_dir}' # in case cwd is below root_dir level
MODEL_DIR = '/home/hmei7/workspace/tcr/gpt_ll/models_ft_1/rita_m/checkpoint-6400'
DEVICE='cuda:0'


def compute_log_likelihood(generated_tcr, model, tokenizer, device):
	model.to(device)
	encoded_input = tokenizer(generated_tcr, return_tensors="pt", padding=True, truncation=True)

	# Move the input tensors to the specified device
	encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

	# Feed the encoded input to the GPT model
	with torch.no_grad():
		outputs = model(**encoded_input)

	# Retrieve the logits (raw outputs) from the GPT model
	logits = outputs.logits

	# Initialize variables to store total log likelihood and sequence length
	total_log_likelihood = 0.0
	sequence_length = len(generated_tcr)

	# Iterate over each amino acid position in the generated TCR sequence
	for i, amino_acid in enumerate(generated_tcr):
		log_likelihood = logits[0, i, tokenizer.convert_tokens_to_ids(amino_acid)]
		total_log_likelihood += log_likelihood

	# Compute the average log likelihood
	average_log_likelihood = total_log_likelihood / sequence_length

	return total_log_likelihood.cpu().numpy()#, average_log_likelihood


def evaluate(folder, bap):
	model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True)
	tokenizer = AutoTokenizer.from_pretrained('/home/hmei7/workspace/tcr/gpt_ll/models_RITA_add_tokens/RITA_m')
	
	# Add special tokens (if not already added during training)
	special_tokens_dict = {'eos_token': '<EOS>', 'pad_token': '<PAD>', 'additional_special_tokens': ['$','<tcr>','<eotcr>','<epi>','<epepi>']}
	num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
	model.resize_token_embeddings(len(tokenizer)) 
	tokenizer.pad_token = tokenizer.eos_token
	# Read the generated TCRs
	generated = Path(PREFIX).joinpath(f'bap_attack/{folder}/attack_{bap}.csv')
	data_file = str(generated)
	designed_TCRs = pd.read_csv(data_file)
	
	log_likelihoods = [999] * len(designed_TCRs)
	for i, designed_tcr in tqdm(enumerate(designed_TCRs['TCRs'])):
		if designed_TCRs['yhat'][i] > 0 and \
			designed_TCRs['TCRs'][i] != 'WRONGFORMAT' and \
			'<' not in designed_TCRs['TCRs'][i] and \
			'>' not in designed_TCRs['TCRs'][i]:
			log_likelihoods[i] = compute_log_likelihood(designed_tcr, model, tokenizer, DEVICE)
	
	
	# Add the max score as a new column in the DataFrame
	designed_TCRs['gpt_ll'] = log_likelihoods
	designed_TCRs.to_csv(data_file, header=True, index=False, float_format='%.5f')
	

if __name__ == '__main__':
	args = parser.parse_args()
	evaluate(args.folder, args.bap)
	# print('Done')