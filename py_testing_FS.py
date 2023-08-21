
import py_mimo
import numpy as np
import tensorflow as tf
import scipy.io as spio
import os
from collections import Counter 
from scipy.stats import mode

tf.reset_default_graph()
tf.set_random_seed(seed=np.random.randint(1,50))
np.random.seed(seed=np.random.randint(1,50))

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                Parameters
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Nt = 16
Nr = 16
L = 30
N = 2*Nt
mod_scheme = "16QAM"
#SNR_vec = [0,2,4,6,8,10,12] # dB
SNR_vec = [12, 14] # dB

test_batch_size = 10000

    
directory_data = "./model/" + str(Nt) + "x" + str(Nr) + "_" + mod_scheme
if not os.path.exists(directory_data):
    os.makedirs(directory_data)
    
directory_model = directory_data
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                            Testing DetNet
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    
  sess.run(init)
  BER_vec = []
  
  for ss in range(len(SNR_vec)):
      snr_test = SNR_vec[ss]
      print('snr_test = ', snr_test, 'dB')      
      ber_file_name = directory_data + "/" + "ber" +  ".mat"
      dat_file_name = directory_data + "/" + str(snr_test) + "dB" +  ".mat"
      
      # Generate testing data
      batch_HY, batch_HH, batch_S, batch_Y, batch_H, batch_N, batch_X_ZF, batch_X_0 = py_mimo.gen_data_ZF(test_batch_size, Nt, Nr, snr_test, mod_scheme, 0, 0)
      
      # Load meta graph and restore weights
      model_file_name = directory_model + "/" + str(snr_test) + "dB/"
      saver = tf.train.import_meta_graph(model_file_name  + "trained_model.meta")
      saver.restore(sess, tf.train.latest_checkpoint(model_file_name))
      
      # Access and load variables
      graph = tf.get_default_graph()
      HY = graph.get_tensor_by_name("HY:0")
      HH = graph.get_tensor_by_name("HH:0")
      X = graph.get_tensor_by_name("X:0")
      X_0 = graph.get_tensor_by_name("X_0:0")
      X_ZF = graph.get_tensor_by_name("X_ZF:0")
      x_hat = graph.get_tensor_by_name("x_hat:0")
      feed_dict={HY: batch_HY, HH: batch_HH, X: batch_S, X_ZF: batch_X_ZF, X_0: batch_X_0}
      
      # Estimate s_hat and its constellations
      s_hat = sess.run(x_hat, feed_dict)
          
      BER = py_mimo.compute_BER(s_hat, batch_S, Nt, mod_scheme)
      print('=====> BER = ', BER)
      
      # save data
      spio.savemat(dat_file_name, {"s":batch_S, "H":batch_H, "y":batch_Y, "n": batch_N, "s_nn":s_hat})
      BER_vec.append(BER)
      
  # save BER results    
  spio.savemat(ber_file_name, {"BER_vec":BER_vec})
