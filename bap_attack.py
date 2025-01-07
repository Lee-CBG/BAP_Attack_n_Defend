from pathlib import Path
import shutil
import os,sys
import json
from tqdm import tqdm
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import subprocess
# from itertools import islice

import torch
import torch.utils
from torch.optim.lr_scheduler import LambdaLR
from transformers import GPT2Tokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model, RewardTrainer, RewardConfig

from utils.rlhf_utils import create_PreferenceDB, create_ScoreQuery, create_RewardInferene, tabula_rasa_RMDataset, lookup_best_reward
from utils.rlhf import create_ppo_config, eval_filter
from utils.data_utils import save_to_csv, override_csv, merge_results, select_candidates, augment_dataset

import warnings
warnings.filterwarnings("ignore")


# os.environ['WANDB_PROJECT'] = 'RLHF4TCRGen'
FILE_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = 'attack' # set root repo as 'attack' to accomendate all plugin reward repos
prefix = str(FILE_PATH)[:(str(FILE_PATH).find(root_dir)+len(root_dir))] if str(FILE_PATH).find(root_dir)!=-1 else str(FILE_PATH)+f'/{root_dir}' # in case cwd is below root_dir level
sys.path.append(prefix)

LOG_DIR = str(FILE_PATH.joinpath('logs'))
CMD_MAP = {'atmTCR': f'conda run -n atm-tcr python bap_reward/atmTCR_infer.py',\
	   'ergo': f'conda run -n ergo python bap_reward/ergo_infer.py',\
	   'pite': f'conda run -n tf26 python bap_reward/pite_infer.py',\
	   'embedder': f'conda run -n torch14_conda python bap_reward/embedder.py',\
}
TRAIN_MAP = {'pite': f'conda run -n tf26 python bap_train/pite_train.py',\
			 'atmTCR': f'conda run -n atm-tcr python bap_train/atmTCR_train.py',\
			 'ergo': f'conda run -n ergo python bap_train/ergo_train.py',\
}
SPECIAL_TOKENS = ["<PAD>", "<tcr>", "<eotcr>", "[CLS]", "[BOS]", "[MASK]", "[SEP]", "<epi>", "<eoepi>", "$", '<unk>']


def collator(data):
	return dict((key, [d[key] for d in data]) for key in data[0])

def lm_generation(trainer, tokenizer, kwargs, data_Path, round, init=False):
	# might need more data for the first round generation
	if init:
		epoch = kwargs.pre_train_steps
	else:
		epoch = kwargs.stepwise_steps
	for _ in tqdm(range(epoch)):
		for batch in tqdm(trainer.dataloader):
			queries = batch["input_ids"]
			response_tensors = []
			for query in queries:
				response = trainer.generate(query, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
											 **kwargs.rl_model.comparison_kwargs)
				for j in range(response.shape[0]):
					response_tensors.append(response[j,:])
			batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
			pairs = [response.replace(' ', '').split('<EOS>')[0].split('$') for response in batch["response"]]
			# epis, tcrs = zip(*pairs)
			info = {'Epitopes': [pair[0] for pair in pairs], 'TCRs': [pair[1] for pair in pairs]}
			save_to_csv(info, round, data_Path)

def rlhf_init(cfg, ppo_trainer, tokenizer):
	rlhf_settings = cfg.rlhf
	device = torch.device(f'cuda:{cfg.device}' if torch.cuda.is_available() else 'cpu')

	if not os.path.exists(rlhf_settings.FILE.RM):
		# Initialize reward model from scratch
		print('Checking Step4:\t No Reward model found...\n')
		# Check if Preference data has been constructed.
		print('====================DETECTING REWARD TRAINING DATASET======================\n')
		if not os.path.exists(FILE_PATH.joinpath(f'data_rlhf/rmDataset.csv')):
			print('Checking Step3:\t No Preference dataset constructed...\n')
			# Check if the batch reward training data has been labeled
			if not os.path.exists(f'{FILE_PATH}/data_rlhf/attack_{cfg.attack.name}_output.csv'):
				print('Checking Step2:\t Reward training data not Labeled...\n')
				# Check if the batch reward training data has been generated
				if not os.path.exists(f'{FILE_PATH}/data_rlhf/attack_{cfg.attack.name}.csv'):
					print('Checking Step1:\t No BATCH DATA for reward model training...\n')
					print('====================GENERATING======================\n')
					
					print('Working Step1:\t Generating BATCH DATA for Reward model fitting...\n')
					lm_generation(ppo_trainer, tokenizer, rlhf_settings, FILE_PATH.joinpath(f'data_rlhf/attack_{cfg.attack.name}.csv'), rd, init=True)
					print(f"Step1 Finshed. BATCH DATA generated @{FILE_PATH.joinpath(f'data_rlhf/attack_{cfg.attack.name}.csv')}! \n")
					rd+=1
				print(f'Working Step2:\t Generating Prefrence Score from BAP:{cfg.attack.name}...')
				if cfg.attack.name in ['catelmp-mlp, pite']:
					pad =  True if cfg.attack.name == 'pite' else False
					embed_command = CMD_MAP['embedder'] + f' --pad {pad} --folder data_rlhf --device {cfg.device}'
					embed_process = subprocess.Popen(embed_command, shell=True, stdout=subprocess.PIPE)
					embed_process.communicate()
				reward_command = CMD_MAP[cfg.attack.name] + f' data_rlhf --device {cfg.device}'
				process = subprocess.Popen(reward_command, shell=True, stdout=subprocess.PIPE)
				stdout, stderr = process.communicate()
				merge_results(FILE_PATH.joinpath(f'data_rlhf/attack_{cfg.attack.name}.csv'))
				print(f"Step2 Finshed. Reward training data labeled @{FILE_PATH.joinpath(f'data_rlhf/attack_{cfg.attack.name}_output.csv')}! \n")
			print(f'Working Step3:\t Gnerating Tabula rasa Preference DATASET construction with {cfg.attack.name}...')
			tabula_rasa_RMDataset(FILE_PATH.joinpath(f'data_rlhf/attack_{cfg.attack.name}_output.csv'), K=rlhf_settings.K)
			print(f"Step3 Finshed. Preference dataset constructed @{FILE_PATH.joinpath('data_rlhf/rmDataset.csv')}! \n")
			# construct and train reward model
			# for named, param in reward_model.named_parameters():
			#	 param.requires_grad = False
			# reward_model.v_head.summary.weight.requires_grad = True
			# reward_model.v_head.summary.bias.requires_grad = True
			# torch.autograd.set_detect_anomaly(True)
		print('Working Step4:\t Training Reward Model from scratch...\n')
		dataset_Reward = create_PreferenceDB(FILE_PATH.joinpath(f'data_rlhf/rmDataset.csv'), tokenizer, device)
		print('Working Step4:\t Generate preference data...\n')
			# weight_decay=0.1
		reward_model = AutoModelForSequenceClassification.from_pretrained(rlhf_settings.FILE.LM, num_labels=1).to(device)
		config_Reward = RewardConfig(output_dir=rlhf_settings.FILE.RM, logging_dir=LOG_DIR, **rlhf_settings.reward_config)
		reward_model.config.pad_token_id = tokenizer.pad_token_id
		reward_Trainer = RewardTrainer(
			args=config_Reward,
			model=reward_model,
			tokenizer=tokenizer,
			train_dataset=dataset_Reward,
			)
		reward_Trainer.train()
		print(f"Step4 Finshed. Reward model trained and saved @{rlhf_settings.FILE.RM}\n")
	else:
		print(f'Loading Reward model @{rlhf_settings.FILE.RM}...\n')
		reward_model = AutoModelForSequenceClassification.from_pretrained(lookup_best_reward(rlhf_settings.FILE.RM), num_labels=1).to(device)
		config_Reward = RewardConfig(output_dir=rlhf_settings.FILE.RM, logging_dir=LOG_DIR, **rlhf_settings.reward_config)
		reward_model.config.pad_token_id = tokenizer.pad_token_id
		reward_Trainer = RewardTrainer(
			args=config_Reward,
			model=reward_model,
			tokenizer=tokenizer,
			)
	return reward_model, reward_Trainer


def bap_init(cfg):
	device = cfg.device
	# retrain bap_model if its not saved at model_dir
	if OmegaConf.is_missing(cfg.attack, 'retrain_model'):
		model_file = f'{cfg.attack.name}_{cfg.data_split}_retrain.{cfg.attack.model_suffix}'
	else:
		model_file = cfg.attack.retrain_model
	if not os.path.exists(FILE_PATH.joinpath(model_file)):
		print('====================DETECTING BAP RETRAINED MODEL======================\n')
		print('Checking Step1:\t No BAP retrained model found...\n')
		print('====================RETRAINING======================\n')
		train_command = TRAIN_MAP[cfg.attack.name] + f" -d {FILE_PATH.joinpath(f'data/{cfg.data_split}_split')} -o {FILE_PATH} --new_model {model_file}  --old_model '' --device {device}"
		train_process = subprocess.Popen(train_command, shell=True, stdout=subprocess.PIPE)
		train_process.communicate()
	else:
		print(f'Loading BAP retrained model @{FILE_PATH.joinpath(model_file)}...\n')
	child_file = Path(model_file).name
	output_dir = Path(cfg.output_dir)
	cur_model = output_dir.joinpath(child_file)
	shutil.copy(FILE_PATH.joinpath(model_file), output_dir.joinpath(child_file))
	print(f'Moved BAP to current experiment directory @{cur_model}...\n')
	return cur_model


def bap_update(cfg, old_model, data):
	# new model should be saved at the same directory as the old model and add one more round number
	device = cfg.device
	old_path = Path(old_model)
	rd = old_path.stem.split('_')[-1]
	if rd.isnumeric():
		new_model = old_path.parent.joinpath(old_path.stem.replace(rd, str(int(rd)+1))).with_suffix(old_path.suffix)
		
	else: 
		new_model = old_path.parent.joinpath(old_path.stem+'_1').with_suffix(old_path.suffix)

	# update bap model with new data
	train_command = TRAIN_MAP[cfg.attack.name] + f' -d {FILE_PATH.joinpath(data)} --output {FILE_PATH} --new_model {new_model} --old_model {old_model} --device {device}'
	train_process = subprocess.Popen(train_command, shell=True, stdout=subprocess.PIPE)
	train_process.communicate()
	return new_model


@hydra.main(config_path='configs', config_name='config', version_base='2.1')
def main(cfg: DictConfig):
	# temporay PPO training setup /* use class registration to avoid this */
	# there is no reason for multiprocess training at this stage
	# cfg = OmegaConf.to_container(cfg)
	rd = 0
	rlhf_settings = cfg.rlhf
	output_dir = cfg.output_dir.replace(cfg.base_dir, '')[1:]
	candidate_dir = f'{output_dir}/iter_0'
	tmp_dir =  f'{output_dir}/tmp'
	candidate_path = FILE_PATH.joinpath(f'{candidate_dir}')
	# NOTE: set the device for PPO training is not easy in this trl version. Using a pratical way here
	os.environ['CUDA_VISIBLE_DEVICES']=str(cfg.device)
	DEVICE = torch.device(f'cuda:{cfg.device}')

	config_PPO = create_ppo_config(
		name=cfg.rlhf.FILE.LM,
	 	configs=rlhf_settings.rl_model.configs, # increase cliprange to allow larger updates
	)
	model = AutoModelForCausalLMWithValueHead.from_pretrained(config_PPO.model_name)
	model_ref = create_reference_model(model)
	tokenizer = GPT2Tokenizer.from_pretrained(cfg.rlhf.FILE.TKN)
	# TODO: check dataset for epi_split
	dataset = create_ScoreQuery(FILE_PATH.joinpath('data/epi_training.txt'), tokenizer)
	ppo_trainer = PPOTrainer(
		config_PPO, 
		model,
		model_ref,
		tokenizer, 
		dataset=dataset, 
		data_collator=collator)
	# model.config.pad_token = tokenizer.pad_token

	# load original bap training samples to generate preference data
	if rlhf_settings.name == 'preference':
		# Check if there is a trained Reward model
		reward_model, reward_Trainer = rlhf_init(cfg, ppo_trainer, tokenizer)
	elif cfg.rlhf.name == 'bap':
		pass
	else: 
		raise NameError('Invalid RLHF name')

	# Integrate wandb
	if wandb.run:
		wandb_cfg = OmegaConf.to_container(cfg.wandb)
		cfg = OmegaConf.merge(cfg, wandb_cfg)
		wandb.init(project=cfg.project, name=cfg.run_name, config=OmegaConf.to_container(cfg))
	
	print(f'{len(ppo_trainer.dataloader)} BATCHS in each EPOCH\n')
	# Check if we need to train a new bap from scratch
	if cfg.attack.re_train:
		model_dir = bap_init(cfg)
	else:
		# TODO: implement if test other model's adaptibility
		model_dir = cfg.attack.default_model

	for r in range(1, cfg.augment_rounds+1):
		# TODO: deal with tmp and result directory if they exist (add a protection layer)
		
		# NOTE: reset Language model after each round of attack
		model = AutoModelForCausalLMWithValueHead.from_pretrained(config_PPO.model_name)
		model_ref = create_reference_model(model)
		tokenizer = GPT2Tokenizer.from_pretrained(cfg.rlhf.FILE.TKN)
		# TODO: check dataset for epi_split
		dataset = create_ScoreQuery(FILE_PATH.joinpath('data/epi_training.txt'), tokenizer)
		ppo_trainer = PPOTrainer(
			config_PPO, 
			model,
			model_ref,
			tokenizer, 
			dataset=dataset, 
			data_collator=collator)
		
		for epoch in tqdm(range(rlhf_settings.generation_rounds)):
			# islice(ppo_trainer.dataloader, 1)
			for batch in ppo_trainer.dataloader:
				query_tensors = batch["input_ids"]
				response_tensors = []
				raw_pairs = []
				for query in query_tensors:
					response = ppo_trainer.generate(query, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, 
										**rlhf_settings.rl_model.generation_kwargs)
					response_tensors.append(response.squeeze()[len(query):])
					raw_pairs.extend([tokenizer.decode(rs).replace(' ', '') for rs in response])
				pairs = [rs.split('<EOS>')[0].split('$') for rs in raw_pairs]

				if rlhf_settings.name == 'preference':
					# update LM
					dataset_pair = create_RewardInferene(raw_pairs , tokenizer, device=DEVICE)
					reward_tensors = reward_model(input_ids=dataset_pair['input_ids'].to(DEVICE), attention_mask=dataset_pair['attention_mask'].to(DEVICE))["logits"].to('cpu')
					stats = ppo_trainer.step(query_tensors, response_tensors, list(torch.unbind(reward_tensors, dim=0)))
					# ppo_trainer.log_stats(stats, batch, reward_tensors.detach().numpy())
					info = {'Epitopes': [p[0] for p in pairs], 'TCRs': [p[1] for p in pairs], 'Rewards': [round(rs.item(),4) for rs in reward_tensors]}
					save_to_csv(info, rd, candidate_path.joinpath(f'attack_{cfg.attack.name}.csv'))

					# Generate bap score (as reference)
					if cfg.attack.name in ['catelmp-mlp', 'pite']:
						pad =  True if cfg.attack.name == 'pite' else False
						embed_command = CMD_MAP['embedder'] + f' --pad {pad} --folder {candidate_dir} --device {cfg.device}'
						embed_process = subprocess.Popen(embed_command, shell=True, stdout=subprocess.PIPE)
						embed_process.communicate()

					reward_command = CMD_MAP[cfg.attack.name] + f' {candidate_dir} --device {cfg.device}'
					process = subprocess.Popen(reward_command, shell=True, stdout=subprocess.PIPE)
					stdout, stderr = process.communicate()

					# update RM
					lm_generation(ppo_trainer, tokenizer, rlhf_settings, candidate_dir+f'/attack_{cfg.attack.name}.csv', rd)
					if cfg.attack.name in ['catelmp-mlp', 'pite']:
						embed_command = CMD_MAP['embedder'] + f' --model {cfg.attack.name} --data {tmp_dir}/attack_{cfg.attack.name}.csv --device {cfg.device}'
						embed_process = subprocess.Popen(embed_command, shell=True, stdout=subprocess.PIPE)
						embed_process.communicate()
					reward_command = CMD_MAP[cfg.attack.name] + f' -d {tmp_dir} -m {model_dir} --device {cfg.device}'
					process = subprocess.Popen(reward_command, shell=True, stdout=subprocess.PIPE)
					stdout, stderr = process.communicate()
					merge_results(FILE_PATH.joinpath(tmp_dir).joinpath(f'attack_{cfg.attack.name}.csv'))
					tabula_rasa_RMDataset(FILE_PATH.joinpath(tmp_dir).joinpath(f'attack_{cfg.attack.name}_output.csv'), K=cfg.rlhf.K)
					dataset_Reward = create_PreferenceDB(FILE_PATH.joinpath(tmp_dir).joinpath(f'rmDataset.csv'), tokenizer, DEVICE)
					reward_Trainer.train_dataset = dataset_Reward
					reward_Trainer.train()
					# remove tmp log directory
					# os.rmdir(str(FILE_PATH.joinpath('tmp')))
				elif rlhf_settings.name == 'bap':
					# update LM
					info = {'Epitopes': [p[0] for p in pairs], 'TCRs': [p[1] for p in pairs]}
					# TODO: check if there is any way to improve this eception handling
					info['TCRs'] = ['WRONGFORMAT' if (not s or any(token in s for token in SPECIAL_TOKENS)) else s for s in info['TCRs']]
					override_csv(info, rd, candidate_path.joinpath(f'attack_{cfg.attack.name}.csv'))

					if cfg.attack.name in ['catelmp-mlp', 'pite']:
						embed_command = CMD_MAP['embedder'] + f' --model {cfg.attack.name} --data {candidate_dir}/attack_{cfg.attack.name}.csv --device {cfg.device}'
						embed_process = subprocess.Popen(embed_command, shell=True, stdout=subprocess.PIPE)
						embed_process.communicate()
					reward_command = CMD_MAP[cfg.attack.name] + f' -d {candidate_dir} -m {model_dir} --device {cfg.device}'
					process = subprocess.Popen(reward_command, shell=True, stdout=subprocess.PIPE)
					rewards_json, _ = process.communicate()
					rewards_data = json.loads(rewards_json)
					reward_tensors = [torch.tensor(value[0], dtype=torch.float32) for value in rewards_data]
					stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
					# wandb.log({
					# 		'iteration': rd,
					# 		# 'loss': stats['loss'],
					# 		'reward': sum(reward_tensors).data/len(reward_tensors)})
					# ppo_trainer.log_stats(stats, batch, eward_tensors.detach().numpy())
				else:
					raise NameError('Invalid RLHF name')
				# filte the generated data and construct new dataset for bap retrain
				eval_filter(folder=f'{candidate_path}', bap=cfg.attack.name, **cfg.attack.filter)
				merge_results(candidate_path.joinpath(f'attack_{cfg.attack.name}.csv'))
				rd+=1
			print(f'\nEpoch {epoch}:')

		# construct new	round of rhlf data generation
		select_candidates(candidate_path.joinpath(f'attack_{cfg.attack.name}_output.csv'), method='neg_control')
		if cfg.attack.name in ['catelmp-mlp', 'pite']:
			embed_command = CMD_MAP['embedder'] + f' --model {cfg.attack.name} --data {candidate_dir}/neg_control.csv --device {cfg.device}'
			embed_process = subprocess.Popen(embed_command, shell=True, stdout=subprocess.PIPE)
			embed_process.communicate()
		# retrain the model with augmented data
		augment_dataset(cfg.attack.name, candidate_path, FILE_PATH.joinpath('data/tcr_split'))
		model_dir = bap_update(cfg, model_dir, candidate_path)
		# evaluate the new model and report results
		# clean_result(storage_path.joinpath(f'{cfg.output_dir}'))
		print(f'Agumentation round: {r}')
		candidate_dir = str((f'{output_dir}/iter_{r}'))
		candidate_path = FILE_PATH.joinpath(f'{candidate_dir}')
	wandb.finish()


if __name__ == '__main__':
	main()