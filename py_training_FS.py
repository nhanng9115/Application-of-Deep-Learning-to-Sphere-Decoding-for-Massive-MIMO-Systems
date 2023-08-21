# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 11:58:21 2019

@author: Nguyen Nhan
"""
import tensorflow as tf

import py_mimo
import numpy as np
import os
tf.reset_default_graph()
tf.set_random_seed(seed=np.random.randint(1,50))
np.random.seed(seed=np.random.randint(1,50))
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                Parameters
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# System parameters
hard_voting = 0
QR = 1

Nt = 16
Nr = 16
N = 2*Nt
L = 30
mod_scheme = "16QAM"
#SNR_vec = [0, 2, 4, 6, 8, 10, 12] # dB
SNR_vec = [10, 12, 14, 16, 18, 20, 22] # dB

# Training parameters
start_learning_rate = 0.0001
decay_factor = 0.97
decay_step_size = 100
train_batch_size = 1000
n_epoch_train = 10000
    
if QR == 0:    
    directory_model = "./model/" + str(Nt) + "x" + str(Nr) + "_" + mod_scheme + "_noQR"
else:
    directory_model = "./model/" + str(Nt) + "x" + str(Nr) + "_" + mod_scheme + "QR"
    
if not os.path.exists(directory_model):
        os.makedirs(directory_model)    
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                        The architecture of DetNet
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def layer(x, n_neuron):
    n_input = int(x.get_shape()[1])
    W = tf.matmul(tf.Variable(tf.random_normal([n_input, n_neuron], stddev=0.01)), tf.eye(n_neuron))
    b = tf.Variable(tf.random_normal([1, n_neuron], stddev=0.01))
    y = tf.matmul(x, W) + b
    return y
    
def network(x, hy, hh, x_0):
    LOSS = []
    x_est = x_0
    t = tf.Variable(0.5)
    alpha = tf.constant(1.0)
    
    for i in range(1,L):
        hhx = tf.squeeze(tf.matmul(tf.expand_dims(x_est, 1), hh), 1) 
        
        # Sparsely connected
        Z = layer(hy - hhx, N) + x_est#layer(x_est, N)
        beta = tf.constant(0.5)

        # Activation fuction
        if mod_scheme == "QPSK":
            x_est = -1 + tf.nn.relu(Z+t)/tf.abs(t) - tf.nn.relu(Z-t)/tf.abs(t)

        else:        
            x_est = -3 + (tf.nn.relu(Z+2+t) - tf.nn.relu(Z+2-t) \
                      + tf.nn.relu(Z+t) - tf.nn.relu(Z-t) \
                      + tf.nn.relu(Z-2+t) - tf.nn.relu(Z-2-t))/tf.abs(t)
                
        dis = tf.reduce_mean(tf.reduce_mean(tf.square(x - x_est), 1))
        # compute cosin similarity
        x_est_norm = tf.nn.l2_normalize(x_est, 1)        
        x_norm = tf.nn.l2_normalize(x, 1)
        cos = 1 - tf.abs(tf.reduce_mean(tf.reduce_mean(tf.multiply(x_est_norm, x_norm), 1)))
        LOSS.append(np.log(i)*( alpha*dis + beta*cos ))        
        
    return x_est, LOSS
 
with tf.device('/gpu:0'):
    HY = tf.placeholder(tf.float32, shape=[None,N], name="HY")
    X = tf.placeholder(tf.float32, shape=[None,N], name="X")
    HH = tf.placeholder(tf.float32, shape=[None,N,N], name="HH")
    X_0 = tf.placeholder(tf.float32, shape=[None,N], name="X_0")
    X_ZF = tf.placeholder(tf.float32, shape=[None,N], name="X_ZF")
    

    x_hat_1, LOSS = network(X, HY, HH, X_0)
    norm_1 = tf.norm(HY - tf.squeeze(tf.matmul(tf.expand_dims(x_hat_1,1),HH),1))
    
    if hard_voting == 1:
        x_hat_2, LOSS = network(X, HY, HH, X_ZF)
        norm_2 = tf.norm(HY - tf.squeeze(tf.matmul(tf.expand_dims(x_hat_2,1),HH),1))
        if tf.less(norm_1, norm_2) is True:
            x_hat = x_hat_1
        else:
            x_hat = x_hat_2
    else:
        x_hat = x_hat_1
        
    x_hat = tf.multiply(x_hat,1,name="x_hat")

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                        Optimization
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
TOTAL_LOSS = tf.add_n(LOSS)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_step_size, decay_factor, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(TOTAL_LOSS)
    
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                            Training DetNet
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=20)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
  
  sess.run(init)
  for ss in range(len(SNR_vec)):
      SNR_dB = SNR_vec[ss]
      DELTA_SNR = 1
      model_file_name = directory_model + "/" + str(SNR_dB) + "dB/" + "trained_model"
      print('SNR_dB = ', SNR_dB, 'dB')
      
      for epoch in range(n_epoch_train + 1):
          with tf.device('/gpu:0'):
              batch_HY, batch_HH, batch_X, _, _, _, batch_X_ZF, batch_X_0 = py_mimo.gen_data_ZF(train_batch_size, Nt, Nr, SNR_dB, mod_scheme, DELTA_SNR, QR)
              feed_dict = {HY: batch_HY, HH: batch_HH, X: batch_X, X_ZF: batch_X_ZF, X_0: batch_X_0}
          
              sess.run(training_op, feed_dict)
          
          if epoch == n_epoch_train:
              
              with tf.device('/gpu:0'):
                  loss, s_hat = sess.run([LOSS[-1], x_hat], feed_dict)

              BER = py_mimo.compute_BER(s_hat, batch_X, Nt, mod_scheme)
              print('Training epoch : ', epoch, ', loss : ', loss, '=====> BER = ', BER)
                  
      
      # Save the trained model for each SNR
      saver.save(sess, model_file_name)
