from datasets import Dataset
import pandas as pd
import torch
import os
from pathlib import Path


def create_PreferenceDB(rmDB_Path, tokenizer, device):
    dataset_Reward = Dataset.from_pandas(pd.read_csv(rmDB_Path))
    # dataset_Reward.set_format("torch", columns=['prompt', "input_ids_chosen", "input_ids_rejected"])
    dataset_Reward.set_format("torch", columns=["chosen", "rejected"])
    dataset_Reward = dataset_Reward.map(
        lambda x: {"input_ids_chosen": tokenizer.encode(x["chosen"], return_tensors="pt")[0, :64].to(device)},
        batched=False,)
    dataset_Reward = dataset_Reward.map( 
        lambda x: {"input_ids_rejected": tokenizer.encode(x["rejected"], return_tensors="pt")[0, :64].to(device)},
        batched=False,)
    dataset_Reward = dataset_Reward.map(
        lambda x: {"attention_mask_chosen": torch.ones(len(x["input_ids_chosen"])).to(device)},
        batched=False,)
    dataset_Reward = dataset_Reward.map(
        lambda x: {"attention_mask_rejected": torch.ones(len(x["input_ids_rejected"])).to(device)},
        batched=False,)
    dataset_Reward = dataset_Reward.remove_columns(['chosen', 'rejected'])
    return dataset_Reward


# def create_PreferenceQuery(bap_TrainigData, tokenizer, device):
#     with open(bap_TrainigData) as f:
#         dataset_line = f.readlines()
#     epis = []
#     tcrs = []
#     for line in dataset_line:
#         epi, tcr = line.split("$")
#         epis.append(epi+'$')
#         tcrs.append(tcr+'<EOS>')
#     epis = list(set(epis))
#     my_dataset = {}
#     my_dataset['epis'] = epis
#     dataset = Dataset.from_dict(my_dataset)
#     dataset.set_format("pytorch")
#     dataset = dataset.map(
#     lambda x: {"input_ids": tokenizer.encode(x["epis"], return_tensors="pt")[0, :64].to(device)},
#     batched=False,
# )
#     dataset = dataset.map(lambda x: {"query": tokenizer.decode(x["input_ids"].to(device))}, batched=False)
#     return dataset


def create_RewardInferene(pairs, tokenizer, device):
    df = pd.DataFrame(pairs, columns=['queries'])
    dataset_Reward = Dataset.from_pandas(df)
    # dataset_Reward.set_format("torch", columns=["input_ids_chosen"])
    dataset_Reward = dataset_Reward.map(
        lambda x: {"input_ids": tokenizer.encode(x["queries"], return_tensors="pt")[0, :64].to(device)},
        batched=False,)
    # dataset_Reward = dataset_Reward.map( 
    #     lambda x: {"attention_mask": tokenizer.encode(x["input_ids"], return_tensors="pt")[0, :64].to(DEVICE)},
    #     batched=False,)
    dataset_Reward = dataset_Reward.remove_columns(['queries'])
    batch_encoded = tokenizer.batch_encode_plus(
    	pairs,
    add_special_tokens=True,
    max_length=72,
    padding='max_length',
    return_attention_mask=True,
    return_tensors='pt'
)
    return batch_encoded

    
def tabula_rasa_RMDataset(bap_TrainigData, K):
    output_path = Path(bap_TrainigData).parent
    contents = pd.read_csv(bap_TrainigData)
    # only process K=2 here
    a = contents.loc[range(0, len(contents), K), :].reset_index(drop=True)
    b = contents.loc[range(1, len(contents), K), :].reset_index(drop=True)
    valid = a[(a['yhat'].values != b['yhat'].values) &
                 (b['TCRs'].apply(lambda x: not isinstance(x,float))) &
                 (a['TCRs'].apply(lambda x: not isinstance(x,float)))
              ].index
    a = a.loc[valid]
    b = b.loc[valid]
    reverse = a[a['yhat']<b['yhat']].index
    accept = a.copy()
    accept.loc[reverse] = b.loc[reverse].copy()
    reject = b.copy()
    reject.loc[reverse] = a.loc[reverse].copy()
    result = pd.concat([accept['Epitopes']+"$"+accept['TCRs']+"<EOS>", reject['Epitopes']+"$"+reject['TCRs']+"<EOS>"], axis=1)
    result.columns = ['chosen', 'rejected']
    # result = pd.concat([accept['Epitopes']+"$", accept['TCRs']+"<EOS>", reject['TCRs']+"<EOS>"], axis=1)
    # result.columns = ['prompt', 'input_ids_chosen', 'input_ids_rejected']
    result.to_csv(f'{output_path}/rmDataset.csv', index=False)
    

def lookup_best_reward(checkpont_dir, best_dir=None):
    if best_dir is None:
        candidates = max([int(sub_dir[sub_dir.find('-'):]) for sub_dir in os.listdir(checkpont_dir)])
        if not candidates:
            raise ValueError('No checkpoint found')
        return Path(checkpont_dir).joinpath(f'checkpoint{candidates}')
    else:
        return Path(checkpont_dir).joinpath(best_dir)


def create_ScoreQuery(bap_TrainigData, tokenizer, device):
    with open(bap_TrainigData) as f:
        dataset_line = f.readlines()
    epis = []
    tcrs = []
    for line in dataset_line:
        epi, tcr = line.split("$")
        epis.append(epi+'$')
        tcrs.append(tcr+'<EOS>')
    epis = list(set(epis))
    my_dataset = {}
    my_dataset['epis'] = epis
    dataset = Dataset.from_dict(my_dataset)
    dataset.set_format("pytorch")
    dataset = dataset.map(
    lambda x: {"input_ids": tokenizer.encode(x["epis"], return_tensors="pt")[0, :64].to(device)},
    batched=False,
)
    dataset = dataset.map(lambda x: {"query": tokenizer.decode(x["input_ids"].to(device))}, batched=False)
    return dataset