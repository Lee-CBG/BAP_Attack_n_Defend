import sys
import time
import os
from pathlib import Path
import random
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from numpy import mean, std
from tensorflow import keras
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score, precision_score, recall_score, f1_score

# import keras
from tensorflow.keras import layers
from keras.layers.merge import concatenate
from tensorflow.math import subtract
from keras.models import Model
from tensorflow.keras.layers import ( 
    Input, Dense, LSTM, concatenate, Flatten, BatchNormalization, Dropout, 
    GlobalMaxPooling1D,TimeDistributed, Bidirectional, LeakyReLU
)
from keras.callbacks import LambdaCallback,TensorBoard,ReduceLROnPlateau, EarlyStopping



FILE_PATH = os.path.dirname(os.path.abspath(__file__))
root_dir = 'attack'
prefix = str(FILE_PATH)[:(str(FILE_PATH).find(root_dir)+len(root_dir))] if str(FILE_PATH).find(root_dir)!=-1 else str(FILE_PATH)+f'/{root_dir}' # in case cwd is below root_dir level
REWARD_PATH = Path(prefix).joinpath('pite')
REWARD_MODEL = '/home/hmei7/workspace/tcr/attack/pite/myTransformer_epi_run_10.h5'


warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

sys.path.append(str(REWARD_PATH))
# with open(REWARD_PATH.joinpath('owndata/owndata').joinpath('model_params.json')) as fp:
# 	default_params = json.load(fp)
args = sys.argv
ATTACK_DATA = Path(prefix).joinpath(f'bap_attack/bap_attack/attack_pite.pkl')
if len(args) > 1:
	ATTACK_DATA = Path(prefix).joinpath(f'bap_attack/{args[1]}/attack_pite.pkl')

class Params:
	epi: str = ''
	n_seq: int = 100   
	# param: dict = field(default_factory=dict)
	# ext: str = 'csv'
# Params.param = default_params


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def get_config(self):
        cfg = super().get_config()
        return cfg
    
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


embed_dim = 1024  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer


def myTransformerNet():
    X_tcr = Input(shape=(22,1024))
    X_epi = Input(shape=(22,1024))
    

    transformer_block_tcr = TransformerBlock(embed_dim, num_heads, ff_dim)
    transformer_block_epi = TransformerBlock(embed_dim, num_heads, ff_dim)
    
    # sentence embedding encoder
    sembed_tcr = transformer_block_tcr(X_tcr)
    sembed_tcr = tf.nn.silu(sembed_tcr)
    sembed_tcr = GlobalMaxPooling1D()(sembed_tcr)
    
    # sembed_epi = embedding_layer(X_epi)
    sembed_epi = transformer_block_epi(X_epi)
    sembed_epi = tf.nn.silu(sembed_epi)
    sembed_epi = GlobalMaxPooling1D()(sembed_epi)
    
    # concate [u, v, |u-v|]
    concate = concatenate([sembed_tcr, sembed_epi, abs(subtract(sembed_tcr,sembed_epi))])
    concate = Dense(1024)(concate)
    concate = BatchNormalization()(concate)
    concate = Dropout(0.3)(concate)
    concate = tf.nn.silu(concate)
    # concate = Dense(1, activation='sigmoid')(concate)
    concate = Dense(1)(concate)
    
    model = Model(inputs = [X_tcr, X_epi], outputs=concate, name='myTransformer_model_256')
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    return model


model = myTransformerNet()
model.load_weights(REWARD_MODEL)   


## Read inputs and process the data
testData = pd.read_pickle(ATTACK_DATA)
X1_test_list, X2_test_list = testData.tcr_embeds.to_list(), testData.epi_embeds.to_list()
X1_test, X2_test = np.array(X1_test_list), np.array(X2_test_list)


## Predict
yhat = model.predict([X1_test, X2_test])

# dat1 = pd.read_csv(f'tmp_epis_tcrs.csv')
data_file = str(ATTACK_DATA)
data_file_original =ATTACK_DATA.parent.joinpath(f'{ATTACK_DATA.stem}'+'.csv')
data_file_output =ATTACK_DATA.parent.joinpath(f'{ATTACK_DATA.stem}_output'+'.csv')
dat1 = pd.read_csv(data_file_original)
dat1['yhat'] = yhat

if not os.path.exists(data_file_output):
	dat1.to_csv(data_file_output, index=False, mode='w')
else:
    dat1.to_csv(data_file_output, header= False, index=False, mode='a')
# Serialize the output as JSON and print it

json_output = json.dumps(yhat.tolist())
print(json_output)