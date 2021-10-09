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
from tensorflow.keras.optimizers import Adam

from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GCSConv, GlobalAvgPool
from spektral.layers.pooling import TopKPool
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.transforms import GCNFilter
from spektral.data import BatchLoader
from spektral.models import GeneralGNN
from spektral.layers import GCNConv, GlobalSumPool
from spektral.layers import AGNNConv
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from lm_split import train, test, validation, label

from sklearn.metrics import confusion_matrix
from groups import group_train, group_test, group_validation








learning_rate = 1e-2  # Learning rate
epochs = 400  # Number of training epochs
es_patience = 0  # Patience for early stopping
batch_size = 32  # Batch size
'''

print('train',len(train))
print('test', len(test))
print('validation', len(validation))
'''
class train_set(Dataset):
    
    def read(self):
        output = []
        directory = './ornet-last-frame-node-features'
        for filename in train:
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



class validation_set(Dataset):

 
    def read(self):
        output = []
        directory = './ornet-last-frame-node-features'
        for filename in validation:
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

class test_set(Dataset):


    def read(self):
        output = []
        directory = './ornet-last-frame-node-features'
        for filename in test:
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

loader_tr = DisjointLoader(dataset_tr, batch_size=3, epochs=100)
loader_va = DisjointLoader(dataset_va, batch_size=1)
loader_te = DisjointLoader(dataset_te, batch_size=1)

m_l =[]

F = dataset_te.n_node_features  # Dimension of node features
n_out = dataset_te.n_labels  # Dimension of the target
#print("n_out",n_out)




X_in = Input(shape=(F,), name="X_in")
A_in = Input(shape=(None,), sparse=True)
I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)


X_1 = GCSConv(10, activation="relu")([X_in, A_in])
X_1, A_1, I_1 = TopKPool(ratio=0.5)([X_1, A_in, I_in])

X_2 = GCSConv(10, activation="relu")([X_1, A_1])
X_2, A_2, I_2 = TopKPool(ratio=0.5)([X_2, A_1, I_1])
X_3 = GCSConv(10, activation="relu")([X_2, A_2])

X_3 = GlobalAvgPool()([X_1, I_1])
output = Dense(n_out, activation="sigmoid")(X_3)


model = Model(inputs=[X_in, A_in, I_in], outputs=output)
opt = Adam(lr=learning_rate)
loss_fn = BinaryCrossentropy()







#@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)

def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)

        #pred_train = predictions.numpy()
        targ = target.astype(np.int64)
        #print(type(pred_train[0][0]))
        loss = loss_fn(target, predictions)#ictions)
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(binary_accuracy(target, predictions))
    return loss, acc


def evaluate(loader):

    actual = []
    guess = []
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        predictions_np = pred.numpy()
        print(predictions_np)
        outs = (
            loss_fn(target, pred),
            tf.reduce_mean(binary_accuracy(target, predictions_np)),
        )
        
        output.append(outs)
        actual.append(target[0][0])
        guess.append(pred[0][0])
        
        
        print('target', target[0][0])
        print('pred', pred.numpy()[0][0])
        '''
        print(type(predictions_np[0][0]))
        print(type(target[0][0]))
        '''
        








        '''
        actual.append(target)
        guess.append(pred)
'''

        '''print('pred',pred)
        print('target', target)'''
        #print(target[0][0])
        #print('{:.4f}',cosine_similarity(target, pred))
        #print(len(output))
    return np.mean(output, 0)








v_loss = []
print("Fitting model")
current_batch = epoch = model_loss = model_acc = 0
best_val_loss = np.inf
best_weights = None
patience = es_patience

for batch in loader_tr:
    outs = train_step(*batch)

    model_loss += outs[0]
    model_acc += outs[1]
    current_batch += 1
    if current_batch == loader_tr.steps_per_epoch:
        model_loss /= loader_tr.steps_per_epoch
        model_acc /= loader_tr.steps_per_epoch
        epoch += 1
        m_l.append(model_loss)


        # Compute validation loss and accuracy
        val_loss, val_acc = evaluate(loader_va)
        v_loss.append(val_loss)
        '''
        print(
            "Ep. {} - Loss: {:.2f} - Acc: {:.2f} - Val loss: {:.2f} - Val acc: {:.2f}".format(
                epoch, model_loss, model_acc, val_loss, val_acc
            )
        )
        '''
        

        # Check if loss improved for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = es_patience
            #print("New best val_loss {:.3f}".format(val_loss))
            best_weights = model.get_weights()
        else:
            patience -= 1
            if patience == 0:
                #print("Early stopping (best val_loss: {})".format(best_val_loss))
                break
        model_loss = 0
        model_acc = 0
        current_batch = 0

################################################################################
# EVALUATE MODEL
################################################################################
print("Testing model")
model.set_weights(best_weights)  # Load best model
test_loss, test_acc = evaluate(loader_te)

print("Done. Test loss: {:.4f}. Test acc: {:.2f}".format(test_loss, test_acc))

'''
e = range(1,101)

plt.plot(e, m_l, 'g', label='Training loss')
plt.plot(e, v_loss, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''
