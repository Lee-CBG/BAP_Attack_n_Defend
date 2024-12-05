import os
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import ( 
	Input, Dense, concatenate, BatchNormalization, 
	Dropout, GlobalMaxPooling1D, LayerNormalization, MultiHeadAttention, Layer
)
# import tensorflow as tf
from tensorflow import keras
from keras.ops import abs, subtract
from keras.callbacks import TensorBoard,ReduceLROnPlateau
# warnings.filterwarnings('ignore')
# warnings.simplefilter(action='ignore', category=FutureWarning)


BATCH_SIZE = 32
SEED = 42
DEVICE = '0'


# dataset is .pkl format
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='/home/hmei7/workspace/tcr/attack/bap_attack/data/tcr_split/training.pkl')
parser.add_argument('-o', '--output_dir', type=str, default='test.h5')
parser.add_argument('--new_model', type=str, default='/home/hmei7/workspace/tcr/attack/bap_attack/model_list')
parser.add_argument('--old_model', type=str, default='')


os.environ["CUDA_VISIBLE_DEVICES"] = f'cuda:{DEVICE}'
np.random.seed(SEED) 
'''
self.list_IDs for healthy_neg
	  train	valid	test
epi   186112   46528	67376
tcr   192063   48016	59937

self.list_IDs for wrong_neg
	  train	valid	test
epi   186112   46528	67376
tcr   191864   47967	60185
'''


class DataGenerator(keras.utils.Sequence):
	def __init__(self, data, nameDir, batch_size=BATCH_SIZE, dim=22, n_channels=1024,
				 n_classes=2, shuffle=True, split=1):
		self.nameDir = nameDir
		self.tmp_data = pd.read_pickle(data)
		assert self.nameDir in ['train', 'valid', 'test'], 'nameDir must be train, valid or test'
		assert split <= 1 and split>=0, 'split is a float between 0 and 1'
		if self.nameDir == 'train' and split <= 1:
			self.data = self.tmp_data.iloc[:int(len(self.tmp_data)*split)]
		elif self.nameDir == 'valid' and split <= 1:
			self.data = self.tmp_data.iloc[int(len(self.tmp_data)*split):]
		elif self.nameDir == 'test':
			self.data = self.tmp_data
		else:
			raise ValueError('Invalid split')
		self.indexes = self.data.index.values
		self.shuffle = shuffle
		self.on_epoch_end()
		
		self.dim = dim
		self.batch_size = batch_size
		
		self.n_channels = n_channels
		self.n_classes = n_classes
		print(f'Length of {self.nameDir} data: {len(self.data)}')

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.data) / self.batch_size))

	def __getitem__(self, index):
		# TODO: Generator cannot be used for indexing. Use inmemory data instead.
		indexes = self.indexes[index*self.batch_size:((index+1)*self.batch_size)]
		x = (np.vstack(self.data['tcr_embeds'][indexes].to_numpy()) , np.vstack(self.data['epi_embeds'][indexes].to_numpy()))
		y = np.vstack(self.data['binding'][indexes])
		return x, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		if self.shuffle == True:
			np.random.shuffle(self.indexes)
			
	def get_dim(self):
		return self.dim
	
	def get_channels(self):
		return self.n_channels

def print_performance(y, yhat):
	print('================Performance========================')
	print('AUC: ' + str(roc_auc_score(y, yhat)))
	yhat[yhat>=0.5] = 1
	yhat[yhat<0.5] = 0

	accuracy = accuracy_score(y, yhat)
	precision1 = precision_score(
		y, yhat, pos_label=1, zero_division=0)
	precision0 = precision_score(
		y, yhat, pos_label=0, zero_division=0)
	recall1 = recall_score(y, yhat, pos_label=1, zero_division=0)
	recall0 = recall_score(y, yhat, pos_label=0, zero_division=0)
	f1macro = f1_score(y, yhat, average='macro')
	f1micro = f1_score(y, yhat, average='micro')

	print('precision_recall_fscore_macro ' + str(precision_recall_fscore_support(y,yhat, average='macro')))
	print('acc is '  + str(accuracy))
	print('precision1 is '  + str(precision1))
	print('precision0 is '  + str(precision0))
	print('recall1 is '  + str(recall1))
	print('recall0 is '  + str(recall0))
	print('f1macro is '  + str(f1macro))
	print('f1micro is '  + str(f1micro))
	return


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
	def __init__(self, embed_dim=1024, num_heads=2, ff_dim=32, name='myTransformer_model_256'):
		super().__init__(name=name)
		self.transformer_block_tcr = TransformerBlock(embed_dim, num_heads, ff_dim)
		self.transformer_block_epi = TransformerBlock(embed_dim, num_heads, ff_dim)
		self.global_max_pool = GlobalMaxPooling1D()
		self.dense1 = Dense(1024)
		self.batch_norm = BatchNormalization()
		self.dropout = Dropout(0.3)
		self.dense2 = Dense(1, activation='sigmoid')
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

def get_model():
	pass

def pite(data, output_dir, new_model, old_model=None):
	print('Start training...')

	# Inputs
	X_tcr = Input(shape=(22, 1024), name='X_tcr')
	X_epi = Input(shape=(22, 1024), name='X_epi')
	model = MyTransformerModel()
	outputs = model([X_tcr, X_epi])
	model = Model(inputs=[X_tcr, X_epi], outputs=outputs)


	traingen = DataGenerator(data, 'train', split=0.8)
	validgen = DataGenerator(data, 'valid', split=0.8)


	# model = MyTransformerModel()
	if old_model:
		model.load_weights(f'{output_dir}/{old_model}')
	# model.load_weights(model_dir)

	logs_path = output_dir + '/logs'
# exp_name = 'myTransformer_'+args.split+'_run_'+str(args.run)
	exp_name = 'myTransformer256_'+'tcr'+'_run_'+str(1)

	early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
												   patience=30, 
												   verbose=1,
												   mode='min',
												  )
	check_point = keras.callbacks.ModelCheckpoint(os.path.join(f'{output_dir}/{new_model}', exp_name + ".keras"),
												  monitor='val_loss', 
												  verbose=1, 
												  save_best_only=True, 
												  mode='min',
												 )
	lrate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
										min_delta=0.0001, min_lr=1e-6, verbose=1)

	callbacks = [check_point, early_stopping, lrate_scheduler]
	model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

	model.fit(traingen, validation_data=validgen, callbacks=callbacks, epochs = 5)
	model.save(os.path.join(f'{output_dir}/{new_model}', exp_name + ".h5"))
	## Evaluation
	print('Evaluating...')
	print('Loading testing data...')
	testData = pd.read_pickle(data)
	testgen = DataGenerator(data, 'test', 1, shuffle=False, batch_size=1)
	yhat = model.predict(testgen)

	## save the result
	y = np.array(testData.binding.to_list())
	print_performance(y, yhat)

	testData['yhat'] = yhat
	testData.to_pickle(f'{output_dir}/{Path(new_model).suffix}_transformer256.pkl')




if __name__ == '__main__':
	args = parser.parse_args()
	pite(args.dataset, args.output_dir, args.new_model, args.old_model)