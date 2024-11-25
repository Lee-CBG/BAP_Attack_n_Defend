from datasets import Dataset
import torch.utils
import inspect
from transformers import AutoTokenizer, pipeline, GPT2Tokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model, RewardTrainer, RewardConfig

from trl.core import LengthSampler
from tqdm import tqdm
from utils.data_utils import save_to_csv, override_csv
import pandas as pd
import subprocess
import json
import torch
# from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import os,sys
import numpy as np
from transformers import DataCollatorWithPadding
import wandb
import hydra

from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig, OmegaConf
from utils.rlhf_utils import create_PreferenceDB, create_ScoreQuery, create_RewardInferene, tabula_rasa_RMDataset, lookup_best_reward
from utils.rlhf import create_ppo_config


# os.environ['WANDB_PROJECT'] = 'RLHF4TCRGen'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
DEVICE = 'cuda:0'
FILE_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = 'attack' # set root repo as 'attack' to accomendate all plugin reward repos
prefix = str(FILE_PATH)[:(str(FILE_PATH).find(root_dir)+len(root_dir))] if str(FILE_PATH).find(root_dir)!=-1 else str(FILE_PATH)+f'/{root_dir}' # in case cwd is below root_dir level
sys.path.append(prefix)

LOG_DIR = str(FILE_PATH.joinpath('log'))
CMD_MAP = {'atm-tcr': f'bash -c "source activate atm_tcr && python {prefix}/bap_attack/bap_reward/atm_tcr.py"',
       'catelmp-mlp': f'bash -c "source activate tf26 && python bap_rewards/pite.py"',\
       'ergo': f'bash -c "source activate torch14_conda && python bap_reward/ergo.py"',\
       'titan': f'conda run -n titan python {prefix}/bap_attack/bap_reward/titan.py',\
	   'pite': f'conda run -n tf26 python bap_reward/pite.py',\
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


@hydra.main(config_path='configs', config_name='config', version_base='2.1')
def main(cfg: DictConfig):
	# temporay PPO training setup /* use class registration to avoid this */
	# there is no reason for multiprocess training at this stage
	# cfg = OmegaConf.to_container(cfg)
	rd = 0
	rlhf_settings = cfg.rlhf

	config_PPO = create_ppo_config(
		name=cfg.rlhf.FILE.LM,
	 	configs=rlhf_settings.rl_model.configs # increase cliprange to allow larger updates
)
	model = AutoModelForCausalLMWithValueHead.from_pretrained(config_PPO.model_name).to(DEVICE)
	model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(config_PPO.model_name).to(DEVICE)
	tokenizer = GPT2Tokenizer.from_pretrained(cfg.rlhf.FILE.TKN)
	dataset = create_ScoreQuery(FILE_PATH.joinpath('data/epi_training.txt'), tokenizer, device=DEVICE)
	ppo_trainer = PPOTrainer(config_PPO, model, model_ref, tokenizer, dataset=dataset, data_collator=collator)
	# model.config.pad_token = tokenizer.pad_token


	# load original bap training samples to generate preference data
	if rlhf_settings.name == 'preference':
        # Check if there is a trained Reward model
		if not os.path.exists(rlhf_settings.FILE.RM):
            # Initialize reward model from scratch
			print('Checking Step4:\t No Reward model found...\n')
            # Check if Preference data has been constructed.
			print('====================DETECTING REWARD TRAINING DATASET======================\n')
			if not os.path.exists(FILE_PATH.joinpath(f'data_bap/rmDataset.csv')):
				print('Checking Step3:\t No Preference dataset constructed...\n')
                # Check if the batch reward training data has been labeled
				if not os.path.exists(f'{FILE_PATH}/data_bap/attack_{cfg.attack.name}_output.csv'):
					print('Checking Step2:\t Reward training data not Labeled...\n')
                    # Check if the batch reward training data has been generated
					if not os.path.exists(f'{FILE_PATH}/data_bap/attack_{cfg.attack.name}.csv'):
						print('Checking Step1:\t No BATCH DATA for reward model training...\n')
						print('====================GENERATING======================\n')
						
						print('Working Step1:\t Generating BATCH DATA for Reward model fitting...\n')
						lm_generation(ppo_trainer, tokenizer, rlhf_settings, FILE_PATH.joinpath(f'data_bap/attack_{cfg.attack.name}.csv'), rd, init=True)
						print(f"Step1 Finshed. BATCH DATA generated @{FILE_PATH.joinpath(f'data_bap/attack_{cfg.attack.name}.csv')}! \n")
						rd+=1

					print(f'Working Step2:\t Generating Prefrence Score from BAP:{cfg.attack.name}...')
					# TODO: check embedding later
					if cfg.attack.name in ['catelmp-mlp, pite']:
						pad =  True if cfg.attack.name == 'pite' else False
						embed_command = f'conda run -n torch14_conda python embedder.py --pad {pad} --folder result'
						embed_process = subprocess.Popen(embed_command, shell=True, stdout=subprocess.PIPE)
						embed_process.communicate()
					reward_command = CMD_MAP[cfg.attack.name] + ' data_bap'
					process = subprocess.Popen(reward_command, shell=True, stdout=subprocess.PIPE)
					stdout, stderr = process.communicate()
					print(f"Step2 Finshed. Reward training data labeled @{FILE_PATH.joinpath(f'data_bap/attack_{cfg.attack.name}_output.csv')}! \n")
				print(f'Working Step3:\t Gnerating Tabula rasa Preference DATASET construction with {cfg.attack.name}...')
				tabula_rasa_RMDataset(FILE_PATH.joinpath(f'data_bap/attack_{cfg.attack.name}_output.csv'), K=rlhf_settings.K)
				print(f"Step3 Finshed. Preference dataset constructed @{FILE_PATH.joinpath('data_bap/rmDataset.csv')}! \n")
    			# construct and train reward model
    			# for named, param in reward_model.named_parameters():
    			#     param.requires_grad = False
    			# reward_model.v_head.summary.weight.requires_grad = True
    			# reward_model.v_head.summary.bias.requires_grad = True
    			# torch.autograd.set_detect_anomaly(True)
			print('Working Step4:\t Training Reward Model from scratch...\n')
			dataset_Reward = create_PreferenceDB(FILE_PATH.joinpath(f'data_bap/rmDataset.csv'), tokenizer, DEVICE)
			print('Working Step4:\t Generate preference data...\n')
				# weight_decay=0.1
			reward_model = AutoModelForSequenceClassification.from_pretrained(rlhf_settings.FILE.LM, num_labels=1).to(DEVICE)
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
			reward_model = AutoModelForSequenceClassification.from_pretrained(lookup_best_reward(rlhf_settings.FILE.RM), num_labels=1).to(DEVICE)
			config_Reward = RewardConfig(output_dir=rlhf_settings.FILE.RM, logging_dir=LOG_DIR, **rlhf_settings.reward_config)
			reward_model.config.pad_token_id = tokenizer.pad_token_id
			reward_Trainer = RewardTrainer(
				args=config_Reward,
				model=reward_model,
				tokenizer=tokenizer,
				)

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
	# start LM and RM model tunning
	for epoch in tqdm(range(rlhf_settings.rounds)):
		for batch in ppo_trainer.dataloader:
			query_tensors = batch["input_ids"]
			response_tensors = []
			raw_pairs = []
			for query in query_tensors:
				response = ppo_trainer.generate(query, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, 
									**rlhf_settings.rl_model.generation_kwargs)
				response_tensors.append(response.squeeze()[len(query):])
				raw_pairs.extend([tokenizer.decode(r).replace(' ', '') for r in response])
			pairs = [r.split('<EOS>')[0].split('$') for r in raw_pairs]

			if rlhf_settings.name == 'preference':
				# update LM
				dataset_pair = create_RewardInferene(raw_pairs , tokenizer, device=DEVICE)
				reward_tensors = reward_model(input_ids=dataset_pair['input_ids'].to(DEVICE), attention_mask=dataset_pair['attention_mask'].to(DEVICE))["logits"].to('cpu')
				stats = ppo_trainer.step(query_tensors, response_tensors, list(torch.unbind(reward_tensors, dim=0)))
				# ppo_trainer.log_stats(stats, batch, reward_tensors.detach().numpy())
				info = {'Epitopes': [p[0] for p in pairs], 'TCRs': [p[1] for p in pairs], 'Rewards': [round(r.item(),4) for r in reward_tensors]}
				save_to_csv(info, rd, FILE_PATH.joinpath(f'result/attack_{cfg.attack.name}.csv'))


				# Generate bap score (as reference)
				if cfg.attack.name in ['catelmp-mlp', 'pite']:
					pad =  True if cfg.attack.name == 'pite' else False
					embed_command = f'conda run -n torch14_conda python embedder.py --pad {pad} --folder result'
					embed_process = subprocess.Popen(embed_command, shell=True, stdout=subprocess.PIPE)
					embed_process.communicate()

				reward_command = CMD_MAP[cfg.attack.name] + ' result'
				process = subprocess.Popen(reward_command, shell=True, stdout=subprocess.PIPE)
				stdout, stderr = process.communicate()

				# update RM
				lm_generation(ppo_trainer, tokenizer, rlhf_settings, FILE_PATH.joinpath(f'tmp/attack_{cfg.attack.name}.csv'), rd)
				if cfg.attack.name in ['catelmp-mlp']:
					pad =  True if cfg.attack.name == 'pite' else False
					embed_command = f'conda run -n torch14_conda python embedder.py --pad {pad} --folder result'
					embed_process = subprocess.Popen(embed_command, shell=True, stdout=subprocess.PIPE)
					embed_process.communicate()
				reward_command = CMD_MAP[cfg.attack.name] + ' tmp'
				process = subprocess.Popen(reward_command, shell=True, stdout=subprocess.PIPE)
				stdout, stderr = process.communicate()
				tabula_rasa_RMDataset(FILE_PATH.joinpath(f'tmp/attack_{cfg.attack.name}_output.csv'), K=cfg.rlhf.K)
				dataset_Reward = create_PreferenceDB(FILE_PATH.joinpath(f'tmp/rmDataset.csv'), tokenizer, DEVICE)
				reward_Trainer.train_dataset = dataset_Reward
				reward_Trainer.train()
				# remove tmp log directory
				# os.rmdir(str(FILE_PATH.joinpath('tmp')))
			elif rlhf_settings.name == 'bap':
				# update LM
				info = {'Epitopes': [p[0] for p in pairs], 'TCRs': [p[1] for p in pairs]}
				# TODO: check if there is any way to improve this eception handling
				info['TCRs'] = ['WRONGFORMAT' if (not s or any(token in s for token in SPECIAL_TOKENS)) else s for s in info['TCRs']]
				override_csv(info, rd, FILE_PATH.joinpath(f'result/attack_{cfg.attack.name}.csv'))
				if cfg.attack.name in ['catelmp-mlp', 'pite']:
					pad =  True if cfg.attack.name == 'pite' else False
					embed_command = f'conda run -n torch14_conda python embedder.py --pad {pad} --folder result'
					embed_process = subprocess.Popen(embed_command, shell=True, stdout=subprocess.PIPE)
					embed_process.communicate()
				reward_command = CMD_MAP[cfg.attack.name] + ' result'
				process = subprocess.Popen(reward_command, shell=True, stdout=subprocess.PIPE)
				rewards_json, _ = process.communicate()
				rewards_data = json.loads(rewards_json)
				reward_tensors = [torch.tensor(value[0], dtype=torch.float32) for value in rewards_data]
				stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
				wandb.log({
						'iteration': rd,
						# 'loss': stats['loss'],
						'reward': reward_tensors/len(reward_tensors)})
				# ppo_trainer.log_stats(stats, batch, eward_tensors.detach().numpy())
			else:
				raise NameError('Invalid RLHF name')
			
			rd+=1

		print(f'\nEpoch {epoch}:')		
	wandb.finish()

if __name__ == '__main__':
	main()