import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', type=str, default='data_bap')
parser.add_argument('-b', '--bap', type=str, default='pite')


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
root_dir = 'attack'
prefix = str(FILE_PATH)[:(str(FILE_PATH).find(root_dir)+len(root_dir))] if str(FILE_PATH).find(root_dir)!=-1 else str(FILE_PATH)+f'/{root_dir}' # in case cwd is below root_dir level
SELECTED_DATA = Path(prefix).joinpath('bap_attack/evaluator/pre_selected_dataset.csv')


def TCRMatch(seq1, seq2):
	# Initialize variables to store the results
	similarity_score = 0.0
	normalization_factor = 0.0

	# Determine the minimum length between seq1 and seq2
	min_length = min(len(seq1), len(seq2))

	# Calculate similarity for each k-mer size (k = 1 to min_length)
	for k in range(1, min_length + 1):
		similarity_kmer = 0.0

		for i in range(len(seq1) - k + 1):
			kmer1 = seq1[i:i+k]
			for j in range(len(seq2) - k + 1):
				kmer2 = seq2[j:j+k]
				kmer_similarity = 1.0  # Initialize k-mer similarity to 1.0

				# Calculate k-mer similarity using BLOSUM62 values
				for p in range(k):
					amino_acid1 = amino_map[kmer1[p]]
					amino_acid2 = amino_map[kmer2[p]]
					kmer_similarity *= blosum62[amino_acid1][amino_acid2]

				similarity_kmer += kmer_similarity

		# Add the k-mer similarity to the overall similarity score
		similarity_score += similarity_kmer
	return similarity_score


def TCRMatchNorm(seq1, seq2):
	normalization_factor = np.sqrt(TCRMatch(seq1, seq1)* TCRMatch(seq2, seq2))
	similarity = TCRMatch(seq1, seq2) / normalization_factor
#	 print("Similarity Score:", similarity)
	return similarity


# BLOSUM62 and amino acid mapping
amino_map = {
	'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 
	'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 
	'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
	'B': 2, 'Z': 5, 'X': 0, '*': 0, '@': 0, 'O': 0,
}

# the matrix that used in TCRMatch
blosum62 = [
	[0.0215, 0.0023, 0.0019, 0.0022, 0.0016, 0.0019, 0.003,
	 0.0058, 0.0011, 0.0032, 0.0044, 0.0033, 0.0013, 0.0016,
	 0.0022, 0.0063, 0.0037, 0.0004, 0.0013, 0.0051],
	[0.0023, 0.0178, 0.002,  0.0016, 0.0004, 0.0025, 0.0027,
	 0.0017, 0.0012, 0.0012, 0.0024, 0.0062, 0.0008, 0.0009,
	 0.001,  0.0023, 0.0018, 0.0003, 0.0009, 0.0016],
	[0.0019, 0.002,  0.0141, 0.0037, 0.0004, 0.0015, 0.0022,
	 0.0029, 0.0014, 0.001,  0.0014, 0.0024, 0.0005, 0.0008,
	 0.0009, 0.0031, 0.0022, 0.0002, 0.0007, 0.0012],
	[0.0022, 0.0016, 0.0037, 0.0213, 0.0004, 0.0016, 0.0049,
	 0.0025, 0.001,  0.0012, 0.0015, 0.0024, 0.0005, 0.0008,
	 0.0012, 0.0028, 0.0019, 0.0002, 0.0006, 0.0013],
	[0.0016, 0.0004, 0.0004, 0.0004, 0.0119, 0.0003, 0.0004,
	 0.0008, 0.0002, 0.0011, 0.0016, 0.0005, 0.0004, 0.0005,
	 0.0004, 0.001,  0.0009, 0.0001, 0.0003, 0.0014],
	[0.0019, 0.0025, 0.0015, 0.0016, 0.0003, 0.0073, 0.0035,
	 0.0014, 0.001,  0.0009, 0.0016, 0.0031, 0.0007, 0.0005,
	 0.0008, 0.0019, 0.0014, 0.0002, 0.0007, 0.0012],
	[0.003,  0.0027, 0.0022, 0.0049, 0.0004, 0.0035, 0.0161,
	 0.0019, 0.0014, 0.0012, 0.002,  0.0041, 0.0007, 0.0009,
	 0.0014, 0.003,  0.002,  0.0003, 0.0009, 0.0017],
	[0.0058, 0.0017, 0.0029, 0.0025, 0.0008, 0.0014, 0.0019,
	 0.0378, 0.001,  0.0014, 0.0021, 0.0025, 0.0007, 0.0012,
	 0.0014, 0.0038, 0.0022, 0.0004, 0.0008, 0.0018],
	[0.0011, 0.0012, 0.0014, 0.001,  0.0002, 0.001,  0.0014,
	 0.001,  0.0093, 0.0006, 0.001,  0.0012, 0.0004, 0.0008,
	 0.0005, 0.0011, 0.0007, 0.0002, 0.0015, 0.0006],
	[0.0032, 0.0012, 0.001,  0.0012, 0.0011, 0.0009, 0.0012,
	 0.0014, 0.0006, 0.0184, 0.0114, 0.0016, 0.0025, 0.003,
	 0.001,  0.0017, 0.0027, 0.0004, 0.0014, 0.012],
	[0.0044, 0.0024, 0.0014, 0.0015, 0.0016, 0.0016, 0.002,
	 0.0021, 0.001,  0.0114, 0.0371, 0.0025, 0.0049, 0.0054,
	 0.0014, 0.0024, 0.0033, 0.0007, 0.0022, 0.0095],
	[0.0033, 0.0062, 0.0024, 0.0024, 0.0005, 0.0031, 0.0041,
	 0.0025, 0.0012, 0.0016, 0.0025, 0.0161, 0.0009, 0.0009,
	 0.0016, 0.0031, 0.0023, 0.0003, 0.001,  0.0019],
	[0.0013, 0.0008, 0.0005, 0.0005, 0.0004, 0.0007, 0.0007,
	 0.0007, 0.0004, 0.0025, 0.0049, 0.0009, 0.004,  0.0012,
	 0.0004, 0.0009, 0.001,  0.0002, 0.0006, 0.0023],
	[0.0016, 0.0009, 0.0008, 0.0008, 0.0005, 0.0005, 0.0009,
	 0.0012, 0.0008, 0.003,  0.0054, 0.0009, 0.0012, 0.0183,
	 0.0005, 0.0012, 0.0012, 0.0008, 0.0042, 0.0026],
	[0.0022, 0.001,  0.0009, 0.0012, 0.0004, 0.0008, 0.0014,
	 0.0014, 0.0005, 0.001,  0.0014, 0.0016, 0.0004, 0.0005,
	 0.0191, 0.0017, 0.0014, 0.0001, 0.0005, 0.0012],
	[0.0063, 0.0023, 0.0031, 0.0028, 0.001,  0.0019, 0.003,
	 0.0038, 0.0011, 0.0017, 0.0024, 0.0031, 0.0009, 0.0012,
	 0.0017, 0.0126, 0.0047, 0.0003, 0.001,  0.0024],
	[0.0037, 0.0018, 0.0022, 0.0019, 0.0009, 0.0014, 0.002,
	 0.0022, 0.0007, 0.0027, 0.0033, 0.0023, 0.001,  0.0012,
	 0.0014, 0.0047, 0.0125, 0.0003, 0.0009, 0.0036],
	[0.0004, 0.0003, 0.0002, 0.0002, 0.0001, 0.0002, 0.0003,
	 0.0004, 0.0002, 0.0004, 0.0007, 0.0003, 0.0002, 0.0008,
	 0.0001, 0.0003, 0.0003, 0.0065, 0.0009, 0.0004],
	[0.0013, 0.0009, 0.0007, 0.0006, 0.0003, 0.0007, 0.0009,
	 0.0008, 0.0015, 0.0014, 0.0022, 0.001,  0.0006, 0.0042,
	 0.0005, 0.001,  0.0009, 0.0009, 0.0102, 0.0015],
	[0.0051, 0.0016, 0.0012, 0.0013, 0.0014, 0.0012, 0.0017,
	 0.0018, 0.0006, 0.012,  0.0095, 0.0019, 0.0023, 0.0026,
	 0.0012, 0.0024, 0.0036, 0.0004, 0.0015, 0.0196]]


def evaluate(folder, bap):
	random.seed(42)
	generated_data = Path(folder).joinpath(f'attack_{bap}.csv')
	generated_data_file = str(generated_data)
	selected_data_file = str(SELECTED_DATA)
	designed_TCRs = pd.read_csv(generated_data_file)
	df = pd.read_csv(selected_data_file)
	df = df[:80]  # Only use positively bind pairs

	num_generated_tcrs = len(designed_TCRs)
	num_preselected_tcrs = len(df) # 50
	scores = np.full((num_generated_tcrs, num_preselected_tcrs), 999)
	designed_TCRs['gpt_ll_mean'] = designed_TCRs['gpt_ll'] / designed_TCRs['TCRs'].str.len()

	# Compute match scores for each pair of generated and preselected real TCRs
	# first check gpt_ll then tcr_match for computational efficiency
	for i, designed_tcr in enumerate(tqdm(designed_TCRs['TCRs'], desc="Computing match scores")):
		if designed_TCRs['yhat'][i] > 0 and \
		   designed_TCRs['gpt_ll_mean'][i] < 1.06 and \
		   designed_TCRs['TCRs'][i] != 'WRONGFORMAT' and \
		   '<' not in designed_TCRs['TCRs'][i] and \
		   'U' not in designed_TCRs['TCRs'][i] and \
		   '>' not in designed_TCRs['TCRs'][i]:
			if len(designed_TCRs['TCRs'][i]) <= 40:
				for j, real_tcr in enumerate(df['tcr']):
					scores[i, j] = TCRMatchNorm(designed_tcr, real_tcr)
			else:
				scores[i, :] = 0
	# Select the maximum score for each generated TCR
	max_scores = np.max(scores, axis=1)
	designed_TCRs['tcr_match'] = max_scores

	# designed_TCRs.drop('gpt_ll_mean', axis=1, inplace=True)
	# Save the updated DataFrame to a CSV file
	designed_TCRs.to_csv(generated_data_file, header= True, index=False, float_format='%.5f')
	

if __name__ == '__main__':
	args = parser.parse_args()
	evaluate(args.folder, args.bap)