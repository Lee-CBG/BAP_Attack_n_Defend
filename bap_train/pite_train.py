import os, sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import ( 
	Input, Dense, concatenate, BatchNormalization, 
	Dropout, GlobalMaxPooling1D, LayerNormalization, MultiHeadAttention, Layer
)
# import tensorflow as tf
from tensorflow import keras
from keras.ops import abs, subtract
from keras.callbacks import ReduceLROnPlateau


cwd = os.getcwd()
root_dir = 'attack'
prefix = cwd[:(cwd.find(root_dir)+len(root_dir))] if cwd.find(root_dir)!=-1	 else cwd+f'/{root_dir}' # in case cwd is below root_dir level

REWARD_DIR = Path(prefix).joinpath('pite')
sys.path.append(str(REWARD_DIR))

BATCH_SIZE = 64
SEED = 42
PRINT_EVERY_EPOCH = 1
DATASET_DIR = Path(prefix).joinpath('/home/hmei7/workspace/tcr/attack/bap_attack/outputs/2024-12-25/05-45-17/result')
OUTPUT_DIR =  Path(prefix).joinpath('bap_attack')

# NEW_MODEL = 'model_list/pite_tcr_retrain.keras'
NEW_MODEL = 'model_list/pite_tcr_retrain_1.keras'
OLD_MODEL = 'model_list/pite_tcr_retrain.keras'
DEVICE = '/GPU:0' if len(tf.config.experimental.list_physical_devices('GPU'))>0 else '/CPU:0'
os.environ["CUDA_VISIBLE_DEVICES"] = f'cuda:{DEVICE}'

np.random.seed(SEED) 


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
	filename = Path(prefix).joinpath(f'bap_attack/logs/{Path(NEW_MODEL).stem}.log')
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=Path, default=DATASET_DIR)
parser.add_argument('-o', '--output_dir', type=Path, default=OUTPUT_DIR)
parser.add_argument('--new_model', type=str, default=NEW_MODEL)
parser.add_argument('--old_model', type=str, default=OLD_MODEL)
parser.add_argument('--epochs', type=int, default=200)


def get_diskid_map(split, dataset):
	# put your retrain model here (very large mapping dataset)
	assert dataset in ['training', 'testing', 'polling'], NameError('Only training, testing, and polling datasets are supported')
	assert split in ['tcr','epi'], NameError('Only tcr and epi splits are supported')
	disk_mapping = '/mnt/disk11/user/pzhang84/data/mappings.csv'
	disk_map = pd.read_csv(disk_mapping)
	disk_map['disk_id'] = disk_map['epi'] + '_' +disk_map['tcr']
	disk_map.set_index('disk_id', inplace=True)
	split_mapping = Path('/home/hmei7/workspace/tcr/attack/bap_attack/data').joinpath(f'{split}_split/{dataset}.pkl')
	split_map = pd.read_pickle(split_mapping)
	split_map['split_id'] = split_map['epi'] + '_' + split_map['tcr']
	return disk_map, split_map['split_id']


class InMemoryDataset(keras.utils.Sequence):
	def __init__(self, dataDir, dataset, batch_size, dim=22, n_channels=1024,
				 n_classes=2, shuffle=True, partial=1):
		self.dataDir = Path(dataDir)
		self.dataset = dataset
		assert partial <= 1 and partial>-1, ValueError('Partial belongs to(-1, 1]')
		assert dataset in ['trainData', 'testData'], NameError('Only trainData', 'testData datasets are supported')
		self.threshold = partial
		index_map = pd.read_pickle(self.dataDir.joinpath(f'{dataset}.pkl'))
		self.points = len(index_map)
		if partial <= 0:
			self.data_mapping = index_map.iloc[int(self.threshold*self.points)-1:self.points]
		else:
			self.data_mapping = index_map.iloc[0:int(self.threshold*self.points)]
		self.indexes = self.data_mapping.index.values
		self.shuffle = shuffle
		self.on_epoch_end()
	
		self.dim = dim
		self.batch_size = batch_size
		self.n_channels = n_channels
		self.n_classes = n_classes
		logger.debug(f'Length of dataset: {len(self.indexes)}')

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.ceil(len(self.indexes) / self.batch_size))

	def __getitem__(self, index):
		# TODO: Generator is used for inDiskDataset
		if index >= len(self):
			raise StopIteration
		indexes = self.indexes[index*self.batch_size:((index+1)*self.batch_size)]
		tcr_list = []
		epi_list = []
		y_list = []
		for i in indexes:
				tcr_list.append(self.data_mapping.loc[i]['tcr_embeds'])
				epi_list.append(self.data_mapping.loc[i]['epi_embeds'])
				y_list.append(self.data_mapping.loc[i]['binding'])
		tcr = np.stack(tcr_list)
		epi = np.stack(epi_list)
		y = np.stack(y_list)
		return (tcr, epi), y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		if self.shuffle == True:
			np.random.shuffle(self.indexes)
		
	def get_dim(self):
		return self.dim
	
	def get_channels(self):
		return self.n_channels


class InDiskDataGenerator(keras.utils.Sequence):
	# this is only used in pite training since its too large to fit in memory
	def __init__(self, split, dataset, batch_size, dim=22, n_channels=1024,
				 n_classes=2, shuffle=True, partial=1):
		self.dataDir = Path('/mnt/disk11/user/pzhang84/data/npzs')
		self.dataset = dataset
		self.split = split
		assert partial <= 1 and partial>-1, ValueError('Partial belongs to(-1, 1]')
		assert dataset in ['training', 'testing', 'pooling'], NameError('Only training, testing, and polling datasets are supported')
		assert split in ['tcr','epi'], NameError('Only tcr and epi splits are supported')
		disk_map, index_map = get_diskid_map(self.split, self.dataset)
		self.threshold = partial
		# self.
		# assert self.nameDir in ['train', 'valid', 'test'], 'nameDir must be train, valid or test'
		# assert 
		self.points = len(disk_map)
		# if self.nameDir == 'train' and split <= 1:
		# 	self.data = 
		if partial <= 0:
			self.data_mapping = index_map.iloc[int(self.threshold*self.points)-1:self.points]
		else:
			self.data_mapping = index_map.iloc[0:int(self.threshold*self.points)]
		self.indexes = self.data_mapping.index.values
		self.disk_map = disk_map
		self.shuffle = shuffle
		self.on_epoch_end()
	
		self.dim = dim
		self.batch_size = batch_size
		self.n_channels = n_channels
		self.n_classes = n_classes
		logger.debug(f'Length of dataset: {len(self.indexes)}')

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.indexes) / self.batch_size))

	def __getitem__(self, index):
		# TODO: Generator is used for inDiskDataset
		indexes = self.indexes[index*self.batch_size:((index+1)*self.batch_size)]
		tcr_list = []
		epi_list = []
		y_list = []
		for i in indexes:
			data = 	None
			# The mappingis ranging from 1 to 300016 but stored as 0 to 300015
			file = self.dataDir.joinpath(f"{self.disk_map.loc[self.data_mapping.loc[i]]['npz_id']-1}.npz")
			with open(file, 'rb') as f:
				data = np.load(f)
				tcr_list.append(data['x_tcr'])
				epi_list.append(data['x_epi'])
				y_list.append(data['y'])
		tcr = np.stack(tcr_list)
		epi = np.stack(epi_list)
		y = np.stack(y_list)
		return (tcr, epi), y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		if self.shuffle == True:
			np.random.shuffle(self.indexes)
		
	def get_dim(self):
		return self.dim
	
	def get_channels(self):
		return self.n_channels
	

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

def pite(data, output_dir, new_model, old_model):
	lr = 0.001
	if not old_model:
		old_model = None
	else:
		lr *= 0.001
		
	logger.info('Start training...')
	# Inputs
	X_tcr = Input(shape=(22, 1024), name='X_tcr')
	X_epi = Input(shape=(22, 1024), name='X_epi')
	
	with tf.device(DEVICE):
		model = MyTransformerModel()
		# model.save
		outputs = model([X_tcr, X_epi])
		model = Model(inputs=[X_tcr, X_epi], outputs=outputs)
		# data split method
		split = data.stem.split('_')[0]
		if old_model is None:
			BATCH_SIZE = 128
		else: 
			BATCH_SIZE = 8
		if old_model is None:
			# in dist generator for training
			traingen = InDiskDataGenerator(split, 'training', BATCH_SIZE, partial=0.8)
			valgen = InDiskDataGenerator(split, 'training', BATCH_SIZE, partial=-0.2)
		else:
			model = load_model(output_dir.joinpath(old_model), custom_objects={'MyTransformerModel': MyTransformerModel})
			traingen = InMemoryDataset(data, 'trainData', BATCH_SIZE, partial=0.8)
			valgen =  InMemoryDataset(data, 'trainData', BATCH_SIZE, partial=-0.2)
		# exp_name = 'myTransformer256_'+'tcr'+'_run_'+str(1)
		new_model_path = Path(new_model)
		early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
													   patience=30, 
													   verbose=1,
													   mode='min',
													  )
		check_point = keras.callbacks.ModelCheckpoint(new_model_path.partent.joinpath(f'{new_model_path.stem}_checkpoint.{new_model_path.suffix}'), 
												      monitor='val_loss', 
													  verbose=1, 
													  save_best_only=True, 
													  mode='min',
													 )
		lrate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
											min_delta=0.0001, min_lr=1e-6, verbose=1)
		callbacks = [check_point, early_stopping, lrate_scheduler]
		optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
		model.compile(loss='binary_crossentropy', optimizer=optimizer)
		model.fit(traingen, validation_data=valgen, callbacks=callbacks, epochs = args.epochs)
	model.save(new_model)
	# model1 = load_model(NEW_MODEL, custom_objects={'MyTransformerModel': MyTransformerModel})
	## Evaluation
	logger.info('Evaluating...')
	if old_model is None:
		testData = InDiskDataGenerator(split, 'testing', BATCH_SIZE, shuffle=False, partial=1)
	else:
		testData = InMemoryDataset(data, 'testData', BATCH_SIZE, shuffle=False, partial=1)
	logger.info(f'Loading data @{data}')
	y = []
	yhat = []
	with tf.device(DEVICE):
		for i in tqdm(range(len(testData))):
			yhat.append(model.predict(tf.convert_to_tensor(testData[i][0]), verbose=0))
			y.append(testData[i][1])
		y = np.concatenate(y)
		yhat = np.concatenate(yhat)
		auc = roc_auc_score(y, yhat)
	logger.info(f'AUC: {auc}')
	yhat[yhat > 0.5] = 1
	yhat[yhat <= 0.5] = 0
	yhat = yhat.flatten().astype(int)
	# print_performance(y, yhat)
	accuracy = accuracy_score(y, yhat)
	logger.info(f'Accuracy: {accuracy}')

if __name__ == '__main__':
	args = parser.parse_args()
	pite(args.dataset, args.output_dir, args.new_model, args.old_model)