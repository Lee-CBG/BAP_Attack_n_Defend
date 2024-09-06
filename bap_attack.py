from datasets import Dataset
from transformers import AutoTokenizer, pipeline, GPT2Tokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import LengthSampler
from tqdm import tqdm
from utils import save_to_csv_1, append_tmp_to_master
import pandas as pd
import subprocess
import json
import torch
from pathlib import Path
import os,sys

BAP_ATTACK = 'titan'
CMD_MAP = {'atm-tcr': f'bash -c "source activate atm_tcr && python bap_reward/atm_tcr.py"',
           'catelmp-mlp': f'bash -c "source activate tf26 && python rewards_bap.py"',\
           'ergo': f'bash -c "source activate torch14_conda && python bap_reward/ergo.py"',\
           'titan': f'bash -c "source activate titan && python bap_reward/titan.py"',\
}


cwd = os.getcwd()
root_dir = 'attack' # set root repo as 'attack' to accomendate all plugin reward repos
prefix = cwd[:(cwd.find(root_dir)+len(root_dir))]
sys.path.append(prefix)

## Configuration
config = PPOConfig(
    # model_name="/mnt/disk07/user/pzhang84/generativeTCR/bap_attack/models_gen/checkpoint-1600",
    model_name = str(Path(prefix).joinpath('bap_attack/models_gen/checkpoint-1600')),
    learning_rate=3e-7, #50x smaller
    batch_size = 128, 
    ppo_epochs = 1,
    adap_kl_ctrl = False,
    init_kl_coef = 1.2,
    steps = 5,
)
# wandb.init(config=config, )


## Load pre-trained GPT2 language models
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = GPT2Tokenizer.from_pretrained(str(Path(prefix).joinpath('bap_attack/models_gen/aa_tokenizer_trained')))
tokenizer.pad_token = tokenizer.eos_token



def load_data(train_path):
    datset = []
    with open(train_path, 'r') as f:
        for line in f:
            datset.append(line.strip()) # remove the newline character at the end of the line
    return datset

def split_line(line):
    epi, tcr = line.split("$")
    return epi+'$', tcr

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])



dataset_line = load_data(Path(prefix).joinpath('bap_attack/data/epi_training.txt'))
epis = []
tcrs = []
for line in dataset_line:
    epi, tcr = split_line(line)
    epis.append(epi)
    tcrs.append(tcr)

epis = list(set(epis))
my_dataset = {}
my_dataset['epis'] = epis
dataset = Dataset.from_dict(my_dataset)
dataset.set_format("pytorch")



dataset = dataset.map(
    lambda x: {"input_ids": tokenizer.encode(x["epis"], return_tensors="pt")[0, :64].to(model.pretrained_model.device)},
    batched=False,
)
dataset = dataset.map(lambda x: {"query": tokenizer.decode(x["input_ids"])}, batched=False)
ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer, dataset=dataset, data_collator=collator)


generation_kwargs = {
    "min_length": -1, # don't ignore the EOS token
    "top_k": 0.0, # no top-k sampling
    "top_p": 1.0, # no nucleus sampling
    "do_sample": True, # yes, we want to sample
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "max_length": 72,
}



## Reward hacking
for epoch in tqdm(range(10)):
    for batch in tqdm(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]

        #### Get response from gpt2 scratch (conda: trl)
        response_tensors = []
        for query in tqdm(query_tensors):
    #         generation_kwargs["max_new_tokens"] = 32
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[len(query):])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]


        #### embeddings: catELMo(tcr, epi) (conda: torch14_conda)
        print('Start embedding')
        epis = [tokenizer.decode(r.squeeze())[:-2] for r in query_tensors]
        tcrs = [response.split('<EOS>')[0] for response in batch["response"]]
        special_tokens = ["<PAD>", "<tcr>", "<eotcr>", "[CLS]", "[BOS]", "[MASK]", "[SEP]", "<epi>", "<eoepi>", "$"]
        tcrs = ['WRONGFORMAT' if (not s or any(token in s for token in special_tokens)) else s for s in tcrs]

        # save_to_csv_1(epis, tcrs, Path(prefix).joinpath(f'bap_attack/log/tmp_epis_tcrs_{BAP_ATTACK}.csv'))
        save_to_csv_1(epis, tcrs, Path(prefix).joinpath(f'bap_attack/log/tmp_epis_tcrs_{BAP_ATTACK}.csv'))

        if BAP_ATTACK in ['catelmp-mlp']:
            embed_command = f'bash -c "source activate torch14_conda && python embedder.py"'
            embed_process = subprocess.Popen(embed_command, shell=True, stdout=subprocess.PIPE)
            embed_process.communicate()


        ### rewards from BAP model (tcr split) (conda: tf26)
        reward_command = CMD_MAP[BAP_ATTACK]
        
        reward_process = subprocess.Popen(reward_command, shell=True, stdout=subprocess.PIPE)
        rewards_json, _ = reward_process.communicate()
        rewards_data = json.loads(rewards_json)
        rewards_bap = [torch.tensor(value[0], dtype=torch.float32) for value in rewards_data]

        rewards = rewards_bap
        append_tmp_to_master()

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
    
    print(f'\nEpoch {epoch}:')
    print(tcrs[:20]) # data peek

    
    
    