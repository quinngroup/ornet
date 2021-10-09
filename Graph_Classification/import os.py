import os
import re
import random
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy,BinaryCrossentropy
from tensorflow.keras.metrics import categorical_accuracy, binary_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from spektral.data import Dataset, DisjointLoader, Graph, BatchLoader
from spektral.layers import GCSConv, GlobalAvgPool
from spektral.layers.pooling import TopKPool , MinCutPool
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.transforms import GCNFilter
from spektral.models import GeneralGNN
from spektral.layers import GCNConv, GlobalSumPool, AGNNConv, GraphSageConv, ECCConv, APPNPConv, ARMAConv, DiffusionConv,GATConv
import matplotlib.pyplot as plt
from lm_split import train_lm, test_lm, validation_lm, label
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
import keras
import time
from helper_functions import pred_to_list, target_to_list, my_round
from sklearn.metrics import classification_report







class train_set(Dataset):
    
    def read(self):
        output = []
        directory = './ornet-last-frame-node-features'
        for filename in train_lm:
            #print(filename)
            #print(type(filename))
            matrix_path = directory +'/'+ filename
            matrix_data = np.load(matrix_path)
            #print(matrix_data[0].shape)
            #print(matrix_data[0])
            adjacency_dimensions = (matrix_data[0].shape[0],matrix_data[0].shape[0])
            adjacency_matrix = np.ones(adjacency_dimensions) # this is a fully connected adjacency matrix with self loops
            target = label(filename)
            #print(target)
            #print(filename)
            #print(target)
            output.append(Graph(x=matrix_data[0], a=adjacency_matrix, y=target))
        random.shuffle(output)
        #print(output)
        return output
class validation_set(Dataset):

 
    def read(self):
        output = []
        directory = './ornet-last-frame-node-features'
        for filename in validation_lm:
            #print(filename)
            #print(type(filename))
            matrix_path = directory +'/'+ filename
            matrix_data = np.load(matrix_path)
            #print(matrix_data[0].shape)
            #print(matrix_data[0])
            adjacency_dimensions = (matrix_data[0].shape[0],matrix_data[0].shape[0])
            adjacency_matrix = np.ones(adjacency_dimensions) # this is a fully connected adjacency matrix with self loops
            target = label(filename)
            #print(filename)
            #fprint(target)
            #print(target)
            output.append(Graph(x=matrix_data[0], a=adjacency_matrix, y=target))
        random.shuffle(output)
        #print(output)
        return output
class test_set(Dataset):


    def read(self):
        output = []
        directory = './ornet-last-frame-node-features'
        for filename in test_lm:
            #print(filename)
            #print(type(filename))
            matrix_path = directory +'/'+ filename
            matrix_data = np.load(matrix_path)
            #print(matrix_data[0].shape)
            #print(matrix_data[0])
            adjacency_dimensions = (matrix_data[0].shape[0],matrix_data[0].shape[0])
            adjacency_matrix = np.ones(adjacency_dimensions) # this is a fully connected adjacency matrix with self loops
            target = label(filename)
            #print(target)
            output.append(Graph(x=matrix_data[0], a=adjacency_matrix, y=target))
        random.shuffle(output)
        #print(output)
        return output




dataset_tr = train_set()
dataset_va = validation_set()
dataset_te = test_set()

loader_tr = BatchLoader(dataset_tr,batch_size =100, epochs = 2000)
loader_va = BatchLoader(dataset_va,batch_size =100)
loader_te = BatchLoader(dataset_te,batch_size =100, epochs = 1)


inputs,target = loader_tr.__next__()
print(inputs[0].shape)
print(inputs[1].shape)

'''

loader_batch = BatchLoader(dataset_tr, batch_size=10)

inputs, target = loader_batch.__next__()
print(len(inputs))

loader_disjoint = DisjointLoader(dataset_tr, batch_size=10)

input2, target2 = loader_disjoint.__next__()



print('length', len(inputs))




print('target[0]', target[0].shape)
print('input[0]', inputs[0].shape)
print(inputs[0])
print(target[0])

'''


F = dataset_tr.n_node_features 
n_out = dataset_tr.n_labels 
learning_rate = 1e-3
es_patience = 10
batch_size = 10

N = max(g.n_nodes for g in dataset_tr)
#epochs = 50





X_in = Input(shape=(F,), name="X_in")
A_in = Input(shape=(None,), sparse=True)
I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)

'''

X_1 = GCSConv(32, activation="relu")([X_in, A_in])
#X_1, A_1, I_1 = TopKPool(ratio=0.5)([X_1, A_in, I_in])
X_2 = GCSConv(32, activation="relu")([X_1, A_in])
#X_2, A_2, I_2 = TopKPool(ratio=0.5)([X_2, A_1, I_1])
#X_3 = GCSConv(32, activation="relu")([X_2, A_2])
X_3 = GlobalSumPool()([X_2])
output = Dense(n_out, activation="sigmoid")(X_3)

'''

X_1 = GCSConv(8, activation="tanh")([X_in, A_in])
X_1, A_1 = MinCutPool(N)([X_1, A_in])
X_2 = GCSConv(8, activation="tanh")([X_1, A_1])
X_2, A_2 = MinCutPool(N)([X_2, A_1])
X_3 = GCSConv(8, activation="tanh")([X_2, A_2])

X_3, A_3 = MinCutPool(N)([X_3, A_2])
X_4 = GCSConv(8, activation="tanh")([X_3, A_3])

X_4, A_4 = MinCutPool(N // 2)([X_4, A_3])
X_5 = GCSConv(10, activation="tanh")([X_4, A_4])
X_5, A_5 = MinCutPool(N // 2)([X_4, A_4])
X_6 = GCNConv(10, activation="tanh")([X_5, A_5])
X_6, A_6 = MinCutPool(N // 2)([X_5, A_5])

X_7 = GlobalSumPool()(X_6)
output = Dense(n_out, activation="sigmoid")(X_7)


# Build model
model = Model(inputs=[X_in, A_in], outputs=output)
opt = Adam(lr=learning_rate)
loss_fn = BinaryCrossentropy()

print(model.summary())

#model.compile(optimizer=opt, loss=loss_fn, metrics=["acc"])

train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
@tf.function(input_signature=loader_tr.tf_signature())
def train_step(inputs, target):
    with tf.GradientTape() as tape:
    	train_acc_metric.reset_states()
    	predictions = model(inputs, training=True)
    	loss = loss_fn(target, predictions) + sum(model.losses)
        #pred = predictions.numpy()
        #print(pred)
        
    gradients = tape.gradient(loss, model.trainable_weights)
    opt.apply_gradients(zip(gradients, model.trainable_weights))
    train_acc_metric.update_state(target, predictions)
    acc = train_acc_metric.result()
    return loss, acc ,predictions, target

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    #val_acc_metric.update_state(y, val_logits)

'''
def my_round(num, cutoff):
	if(num>=cutoff):
		return 1
	else:
		return 0

'''
test_loss = []
epochs = 10
best_weights = None
step = 0
best_accuracy = 0
for batch in loader_tr:
	lss, acc, predictions,target = train_step(*batch)

	print(
            "Training loss (for one batch) at step %d: %.4f"
            % (step, float(lss))
        )
	test_loss.append(float(lss))
	#print('acc', acc)
	#print(predictions)
	#print(target)
	step +=1
	
	guess = pred_to_list(predictions)
	actual = target_to_list(target)
	rnd = my_round(guess)
	print(confusion_matrix(actual,rnd))
	train_accuracy = accuracy_score(actual,rnd)
	print('acc', accuracy_score(actual, rnd))
	if train_accuracy > best_accuracy:
		best_weights = model.get_weights()

	#print(guess)
	#print(actual)
	#print(rnd)


test_preds = []
test_targets = []
@tf.function
def test_step(x, y):
    logits = model(x, training=False)
    return logits, y
step = 0
for batch in loader_te:
	model.set_weights(best_weights)
	#print(step)
	inputs, target = batch

	size = inputs[0].shape[0]
	#print(target.shape)
	'''
	for i in range(16):
		print(inputs[0][i].shape)
	'''
		#print(inputs[0][2].shape)
	#step+=1
	

	for i in range(size):
		test_prediction = model([inputs[0][i], inputs[1][i]], training=False)
		print(test_prediction.numpy()[0][0], target[i][0])
		test_preds.append(test_prediction.numpy()[0][0])
		test_targets.append(target[i][0])

	round_test_preds = my_round(test_preds)
	print(confusion_matrix(round_test_preds, test_targets))
	print('accuracy: ', accuracy_score(round_test_preds,test_targets))
	print(classification_report(round_test_preds,test_targets))

	#print(inputs[0][0].shape, inputs[1][0].shape)

'''

e = range(1,2000)

plt.plot(e, test_loss, 'g', label='Training loss')
#plt.plot(e, v_loss, 'b', label='validation loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''
