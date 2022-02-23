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
X = np.loadtxt('data/X_reg.txt')
Y = np.loadtxt('data/Y_reg.txt')
X = X.astype('float32')
Y = Y.astype('float32')
sample_size = X.shape[0]
all_indices = range(len(Y))
shuffled_indices = tf.random.shuffle(all_indices)
X, Y= tf.gather(X, shuffled_indices), tf.gather(Y, shuffled_indices)
X, Y=X.numpy(), Y.numpy()
X_train_raw, X_test, Y_train_raw, Y_test = train_test_split(X, Y, test_size=0.15)


#parameters setup
model_type = 'regression'
reg_type = 'linear'
L = 10 #number of models
nsample = 100 #number of samples used for approximate expectation in ELBO
nepoch =50 #training epoch
input_size = X.shape[1]
initial_size = X.shape[1]
hidden_sizes = [5]# a list of numbers indicate hidden sizes
lamb = 1.0 #weight parameter in loss function
batch_size = 50 
sigma = 0.0001
temperature = 0.1 #for gumbel-softmax trick
tau = 0.3 #for scale alpha in softmax
mini_loss = np.mean(np.square(Y_test-np.mean(Y_test)))*1.0 #this is used as the threshold for purity
l = 0
iteration = 0
max_iter = 15


#run lasso for init
from sklearn.linear_model import LassoCV
clf = LassoCV(cv=5, random_state=0).fit(X_train_raw, Y_train_raw)
pred_train = clf.predict(X_train_raw)
acc_train = np.mean((Y_train_raw - pred_train)**2)
pred_test = clf.predict(X_test)
acc_test = np.mean((Y_test - pred_test)**2)
print(f'Lasso Regression| train {acc_train:.4f} test {acc_test:.4f}')


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

#model
all_model = list()
init_vals = []
model = ESNN(model_type, reg_type, sigma, input_size, hidden_sizes, temperature, tau, False, init_vals)
all_myloss = list()
all_prbs = list()
all_cs = list()
while l<L and iteration<= max_iter:
    myloss = np.zeros((nepoch, 4))
    for epoch in range(0, nepoch):
        learning_rate = 0.005*(0.995**epoch) # for classification
        model.optimizer = tf.optimizers.Adam(lr = learning_rate)
        all_indices = range(len(Y_train_raw))
        shuffled_indices = tf.random.shuffle(all_indices)
        train_bnn(model, tf.gather(X_train_raw, shuffled_indices), tf.gather(Y_train_raw, shuffled_indices), batch_size, learning_rate, True, nsample, 0.00005, 10.0)#0.00005
        pred, nll, kl = model.call(X_train_raw, Y_train_raw, True, 100)
        temp_train_acc = np.mean(tf.losses.MSE(pred[:,:,0], Y_train_raw))
        pred, temp_test_nll, kl = model.call(X_test, Y_test, True, 100)
        temp_test_acc = np.mean(tf.losses.MSE(pred[:,:,0], Y_test))
        elbo = nll+kl
        myloss[epoch,0] = elbo
        myloss[epoch,1] = temp_train_acc
        # myloss[epoch,2] = temp_val_acc
        myloss[epoch,2] = temp_test_acc
        print("Iteration", iteration)
        print("Train loss", temp_train_acc)
        print("l", l)
        print("mini loss", mini_loss)
        print("Test loss", temp_test_acc)
        prbs = np.asarray(tf.nn.softmax(model.bnn.w_alpha[:,0]/tau))
        print('#################################################################################################################')
        print(np.where(prbs>0.1))
        print(np.where(prbs == np.max(prbs)))
        if epoch>3 and model_type == 'regression' and temp_test_acc<mini_loss-0.05:
            break
        if epoch>10 and model_type == 'regression' and temp_test_acc<mini_loss-0.01:
            break
        if epoch > 30:
            curr_avg = np.max(myloss[epoch-2:epoch,2])
            pre_avg = np.max(myloss[epoch-4:epoch-2,2])
            if model_type == 'regression' and curr_avg>= pre_avg:
                break
    if model_type == 'regression' and myloss[epoch,2]<=mini_loss-0.01:
        mini_loss = myloss[epoch,2]
        l += 1
        all_myloss.append(myloss)
        all_model.append(model)
        temp_prbs = np.asarray(tf.nn.softmax(model.bnn.w_alpha[:,0]/tau))
        if temp_prbs.shape[0]<initial_size:
            if len(all_cs) == 1:
                toinsert = np.unique(all_cs[0])
            else:
                toinsert = np.unique(np.concatenate(all_cs))
            temp_to_add = temp_prbs
            for pos in toinsert:
                temp_to_add = np.insert(temp_to_add, pos, 1e-10)
            all_prbs.append(temp_to_add)
        else:
            temp_to_add = temp_prbs
            all_prbs.append(temp_to_add)
        #derive residuals
        pred, nll, kl = model.call(X_train_raw, Y_train_raw, True, 100)
        res_train = np.mean(pred[:,:,0], axis = 0) - Y_train_raw
        pred, temp_test_nll, kl = model.call(X_test, Y_test, True, 100)
        res_test = np.mean(pred[:,:,0], axis = 0) - Y_test
        Y_train_raw = res_train
        Y_test = res_test
        #compute cs
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
            # all_cs.append(temp_cs)
            #remove found variables
            X_train_raw = np.delete(X_train_raw, temp_cs, axis = 1)
            X_test = np.delete(X_test, temp_cs, axis = 1)
            input_size = X_train_raw.shape[1]
            #add cs with correct idx
            nsnp = temp_to_add.shape[0]
            for temp_j in range(nsnp):
                cs_idx = nsnp-temp_j
                if sum(np.sort(temp_to_add)[cs_idx:])>0.95:
                    break
            temp_cs = np.argsort(temp_to_add)[cs_idx:]
            all_cs.append(temp_cs)
    model = SNN(model_type, reg_type, sigma, input_size, hidden_sizes, temperature, tau, False, init_vals)
    iteration+=1


#write data
all_myloss = np.asarray(all_myloss)
all_prbs = np.asarray(all_prbs)
import pickle
all_data = {}
all_data['loss'] = all_myloss
all_data['pips'] = all_prbs
all_data['cs'] = all_cs

#
#
print(np.where(all_data['pips']>0.5))
print(all_data['cs'])
