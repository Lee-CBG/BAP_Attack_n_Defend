import os, sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
from copy import deepcopy

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import ( 
	Input, Dense, concatenate, BatchNormalization, 
	Dropout, GlobalMaxPooling1D, LayerNormalization, MultiHeadAttention, Layer
)
# import tensorflow as tf
from tensorflow import keras
from keras.ops import abs, subtract


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
root_dir = 'attack'
prefix = str(FILE_PATH)[:(str(FILE_PATH).find(root_dir)+len(root_dir))] if str(FILE_PATH).find(root_dir)!=-1 else str(FILE_PATH)+f'/{root_dir}' # in case cwd is below root_dir level
REWARD_PATH = Path(prefix).joinpath('pite')
REWARD_MODEL = 'outputs/2024-12-26/22-01-39/pite_tcr_retrain.keras'
ATTACK_DATA = 'outputs/2024-12-26/22-01-39/result'
BATCH_SIZE = 128
DEVICE = '/GPU:0' if len(tf.config.experimental.list_physical_devices('GPU'))>0 else '/CPU:0'
sys.path.append(str(REWARD_PATH))
os.environ["CUDA_VISIBLE_DEVICES"] = f'cuda:{DEVICE}'


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--attack_data', type=str, default=ATTACK_DATA)
parser.add_argument('-m', '--model_dir', type=str, default=REWARD_MODEL)


class TransformerBlock(Layer):
	def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
		super(TransformerBlock, self).__init__()
		self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
		self.ffn = keras.Sequential(
			[Dense(ff_dim, activation="relu"), Dense(embed_dim),]
		)
		self.layernorm1 = LayerNormalization(epsilon=1e-6)
		self.layernorm2 = LayerNormalization(epsilon=1e-6)
		self.dropout1 = Dropout(rate)
		self.dropout2 = Dropout(rate)

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


class MyTransformerModel(Model):
	def __init__(self, embed_dim=1024, num_heads=2, ff_dim=32, name='myTransformer_model_256', **kwargs):
		super().__init__(name=name)
		self.transformer_block_tcr = TransformerBlock(embed_dim, num_heads, ff_dim)
		self.transformer_block_epi = TransformerBlock(embed_dim, num_heads, ff_dim)
		self.global_max_pool = GlobalMaxPooling1D()
		self.dense1 = Dense(1024)
		self.batch_norm = BatchNormalization()
		self.dropout = Dropout(0.3)
		# remove activation layer
		self.dense2 = Dense(1)
		# self.activation = Activation('swish')
	
	def call(self, inputs, training=True):
		X_tcr, X_epi = inputs
		sembed_tcr = self.transformer_block_tcr(X_tcr, training=training)
		sembed_tcr = keras.activations.swish(sembed_tcr)
		sembed_tcr = self.global_max_pool(sembed_tcr)

		sembed_epi = self.transformer_block_epi(X_epi, training=training)
		sembed_epi = keras.activations.swish(sembed_epi)
		sembed_epi = self.global_max_pool(sembed_epi)

		# Concatenate embeddings
		concate = concatenate([
			sembed_tcr, 
			sembed_epi, 
			abs(subtract(sembed_tcr, sembed_epi))
		])
		concate = self.dense1(concate)
		concate = self.batch_norm(concate, training=training)
		concate = self.dropout(concate, training=training)
		concate = keras.activations.swish(concate)
		output = self.dense2(concate)
		return output


def pite(attack_data, model_dir):
	model = load_model(Path(prefix).joinpath('bap_attack').joinpath(model_dir), custom_objects={'MyTransformerModel': MyTransformerModel}) 
	data = Path(attack_data).joinpath('attack_pite.pkl')
	## Read inputs and process the data
	testData = pd.read_pickle(data)
	
	X1_test_list, X2_test_list = testData.tcr_embeds.to_list(), testData.epi_embeds.to_list()
	X1_test, X2_test = np.array(X1_test_list), np.array(X2_test_list)

	## Predict
	with tf.device(DEVICE):
		X1_test = tf.convert_to_tensor(X1_test, dtype=tf.float32)
		X2_test = tf.convert_to_tensor(X2_test, dtype=tf.float32)
		yhat = model.predict((X1_test, X2_test))
	
	# dat1 = pd.read_csv(f'tmp_epis_tcrs.csv')
	data_file_original =data.parent.joinpath(f'{data.stem}'+'.csv')
	# data_file_output =ATTACK_DATA.parent.joinpath(f'{ATTACK_DATA.stem}_output'+'.csv')
	dat1 = pd.read_csv(data_file_original)
	dat1['yhat'] = np.round(yhat.squeeze(),5)
	dat1.to_csv(data_file_original, index=False, mode='w')
	json_output = json.dumps(yhat.tolist())
	print(json_output)


if __name__ == '__main__':
    args = parser.parse_args()
    pite(args.attack_data, args.model_dir)