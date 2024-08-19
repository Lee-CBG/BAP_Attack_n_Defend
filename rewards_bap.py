import sys
import time
import os
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from numpy import mean, std
from tensorflow import keras
from tensorflow.math import subtract

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score, precision_score, recall_score, f1_score
from keras.callbacks import EarlyStopping
from keras.layers import Input, Flatten, Dense, Dropout, LeakyReLU
from keras.models import Model
from keras.layers.merge import concatenate
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, LayerNormalization
)

import json

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

parser = argparse.ArgumentParser()
parser.add_argument('--epi', type=str)
parser.add_argument('--n_seq', type=int, default=100)
args = parser.parse_args()


## Define models and load its weights
inputA = Input(shape=(1024,))
inputB = Input(shape=(1024,))

x = Dense(2048,kernel_initializer = 'he_uniform')(inputA)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = tf.nn.silu(x)
x = Model(inputs=inputA, outputs=x)

y = Dense(2048,kernel_initializer = 'he_uniform')(inputB)
y = BatchNormalization()(y)
y = Dropout(0.3)(y)
y = tf.nn.silu(y)
y = Model(inputs=inputB, outputs=y)
combined = concatenate([x.output, y.output])

z = Dense(1024)(combined)
z = BatchNormalization()(z)
z = Dropout(0.3)(z)
z = tf.nn.silu(z)
# z = Dense(1, activation='sigmoid')(z)
z = Dense(1)(z) ## we did not use 'sigmoid' function, because we want to use the logits as the rewards
model = Model(inputs=[x.input, y.input], outputs=z)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')



# model.load_weights('/mnt/disk07/user/pzhang84/embeds_protein/BAP_neg_healthy_repertoires/models/catELMo_4_layers_1024/catELMo_4_layers_1024_epi_seed_42_fraction_1.0.hdf5')
model.load_weights('/mnt/disk07/user/pzhang84/embeds_protein/BAP_neg_wrong_combination/models/catELMo_4_layers_1024_shuffling_negatives/catELMo_4_layers_1024_shuffling_negatives_tcr_seed_42.hdf5')
# model.summary()


## Read inputs and process the data
testData = pd.read_pickle(f"tmp_epis_tcrs.pkl")
X1_test_list, X2_test_list = testData.tcr_embeds.to_list(), testData.epi_embeds.to_list()
X1_test, X2_test = np.array(X1_test_list), np.array(X2_test_list)


## Predict
yhat = model.predict([X1_test, X2_test])

dat1 = pd.read_csv(f'tmp_epis_tcrs.csv')

yhat_list = yhat.tolist()
dat1['yhat'] = yhat_list

# dat1['yhat'] = dat1['yhat'].apply(lambda x: float(x.strip('[]'))) # convert into float
dat1.to_csv(f'tmp_epis_tcrs.csv', index=False)

# Serialize the output as JSON and print it
yhat_list = np.array(yhat_list)
yhat_list.reshape(-1)
yhat_list = yhat_list.tolist()
json_output = json.dumps(yhat_list)
print(json_output)