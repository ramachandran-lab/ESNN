import tensorflow as tf
import numpy as np
import sys
import random
from scipy.stats import norm
import os
from sklearn.model_selection import train_test_split
from ESNN_layers import *
import pickle


#load example data
X = np.loadtxt('data/X_binary.txt')
Y = np.loadtxt('data/Y_binary.txt')
X = X.astype('float32')
Y = Y.astype('float32')
sample_size = X.shape[0]
all_indices = range(len(Y))
shuffled_indices = tf.random.shuffle(all_indices)
X, Y= tf.gather(X, shuffled_indices), tf.gather(Y, shuffled_indices)
X, Y=X.numpy(), Y.numpy()
X_train_raw, X_test, Y_train_raw, Y_test = train_test_split(X, Y, test_size=0.15)


#parameters setup
model_type = 'classification'
reg_type = 'logistic'
L = 10 #number of models
nsample = 100 #number of samples used for approximate expectation in ELBO
nepoch =50 #training epoch
input_size = X.shape[1]
hidden_sizes = [5]# a list of numbers indicate hidden sizes
lamb = 1.0 #weight parameter in loss function
batch_size = 50 
sigma = 0.0001
temperature = 0.1 #for gumbel-softmax trick
tau = 0.3 #for scale alpha in softmax
max_acc = 0.5 #this is used as the threshold for purity, for balanced case, acc of 0.5 is a good starting point as a representation of likelihood. For imbalanced case, use other value
l = 0
iteration = 0
max_iter = 15

#helper
def binary_acc(y, pred):
    return np.where(y == pred)[0].shape[0]/y.shape[0]

#optional: run lasso for initialization 
from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(cv=5, random_state=0, penalty = 'l1', solver = 'saga').fit(X_train_raw, Y_train_raw)
pred_train = clf.predict(X_train_raw)
acc_train = binary_acc(Y_train_raw, pred_train)
pred_test = clf.predict(X_test)
acc_test = binary_acc(Y_test, pred_test)
print(f'Logistic Regression l2| train {acc_train:.4f} test {acc_test:.4f}')







#helper
def get_pips(model, tau):
    all_prbs = list()
    for i in range(len(model.models)):
        prbs = np.asarray(tf.nn.softmax(model.models[i].bnn.w_alpha[:,0]/tau))
        all_prbs.append(prbs)
    all_prbs = np.asarray(all_prbs)
    pips = 1 - np.prod(1-all_prbs, axis = 0)
    return pips



####################################NN
##initializations
#if initialize with lasso coefficient
init_val = np.transpose(abs(clf.coef_))
init_val = init_val.astype('float32')
init_vals = list()
for i in range(L):
    temp_init_val = init_val
    temp_init_val = np.reshape(temp_init_val, (input_size, 1))
    init_vals.append(tf.convert_to_tensor(temp_init_val))
#model
model = JointModel(1, model_type, reg_type, sigma, input_size, hidden_sizes, temperature, tau, init_vals)
all_myloss = list()
#Training
while l<L and iteration<= max_iter:
    myloss = np.zeros((nepoch, 4))
    for epoch in range(0, nepoch):
        learning_rate = 0.01*(0.995**epoch) 
        model.optimizer = tf.optimizers.Adam(lr = learning_rate)
        all_indices = range(len(Y_train_raw))
        shuffled_indices = tf.random.shuffle(all_indices)
        train_bnn_joint(model, tf.gather(X_train_raw, shuffled_indices), tf.gather(Y_train_raw, shuffled_indices), batch_size, learning_rate, True, nsample, 0.00005, 10.0, l)
        if model_type == 'classification':
            logits, probability, nll, kl = model.call(X_train_raw, Y_train_raw, True, 100, l)
            temp_train_acc = accuracy(probability, Y_train_raw)
            logits, probability, temp_test_nll, kl = model.call(X_test, Y_test, True, 100, l)
            temp_test_acc = accuracy(probability, Y_test)
        elbo = nll+kl
        myloss[epoch,0] = elbo
        myloss[epoch,1] = temp_train_acc
        myloss[epoch,2] = temp_test_acc
        print("Iteration", iteration)
        print("Train loss", temp_train_acc)
        print("l", l)
        print("Test loss", temp_test_acc)
        if epoch>3 and model_type == 'classification' and temp_test_acc-max_acc>=0.05:
            break
        if epoch>10 and model_type == 'classification' and temp_test_acc-max_acc>=0.01:
            break
        if epoch > 30:
            curr_avg = np.max(myloss[epoch-2:epoch,2])
            pre_avg = np.max(myloss[epoch-4:epoch-2,2])
            if model_type == 'classification' and curr_avg<=pre_avg:
                break
    if model_type == 'classification':
        if temp_test_acc-max_acc>=0.01:
            max_acc = myloss[:epoch,2][-1]
            l += 1
            all_myloss.append(myloss)

        else:
            del model.models[l]
        if len(model.models)>0:
            pips = get_pips(model, tau)
            pips = np.reshape(pips, (input_size, 1))
            temp_init_val = init_val*(1-pips)
        else:
            temp_init_val = init_val
        model.models.append(BNNSparseMLP(model_type, reg_type, sigma, input_size, hidden_sizes, temperature, tau, True, temp_init_val))
    iteration += 1
#delete last untrained model
del model.models[len(model.models)-1]
#postprocess
all_prbs = list()
for i in range(len(model.models)):
    prbs = np.asarray(tf.nn.softmax(model.models[i].bnn.w_alpha[:,0]/tau))
    all_prbs.append(prbs)
all_myloss = np.asarray(all_myloss)
all_prbs = np.asarray(all_prbs)
all_data = {}
all_data['loss'] = all_myloss
all_data['pips'] = all_prbs


all_cs = list()
for i in range(len(all_data['pips'])):
    temp_prbs = all_data['pips'][i]
    print(np.sum(temp_prbs))
    nsnp = temp_prbs.shape[0]
    for temp_j in range(nsnp):
        cs_idx = nsnp-temp_j
        if sum(np.sort(temp_prbs)[cs_idx:])>0.95:
            break
    temp_cs = np.argsort(temp_prbs)[cs_idx:]
    if temp_cs.shape[0]>1:
        cc = np.corrcoef(np.transpose(X[:,temp_cs]))
        for k in range(cc.shape[0]):
            cc[k][k]=0.5
    else:
        cc = 0.5
    if np.min(cc)>=0.5:
        all_cs.append(temp_cs)
all_data['cs'] = all_cs

#
#
print(np.where(all_data['pips']>0.5))
print(all_data['cs'])





