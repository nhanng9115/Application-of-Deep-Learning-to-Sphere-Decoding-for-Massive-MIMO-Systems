"""
Created on Fri Jan 18 15:48:01 2019

@author: Nguyen Nhan
"""

import numpy as np
import scipy.io as spio
import copy
import scipy.linalg   # SciPy Linear Algebra Library

# Create modulation mapping and demapping table
def bitfield(n):
    return [int(digit) for digit in bin(n)[2:]] # [2:] to chop off the "0b" part 

def get_table(mod_scheme, table_type):
    if mod_scheme == "16QAM":
        sym_set = [-3-3j, -3-1j, -3+3j, -3+1j, -1-3j, -1-1j, -1+3j, -1+1j,\
                   3-3j, 3-1j, 3+3j, 3+1j, 1-3j, 1-1j, 1+3j, 1+1j]
        mapping_table = {
            (0,0,0,0) : -3-3j,
            (0,0,0,1) : -3-1j,
            (0,0,1,0) : -3+3j,
            (0,0,1,1) : -3+1j,
            (0,1,0,0) : -1-3j,
            (0,1,0,1) : -1-1j,
            (0,1,1,0) : -1+3j,
            (0,1,1,1) : -1+1j,
            (1,0,0,0) :  3-3j,
            (1,0,0,1) :  3-1j,
            (1,0,1,0) :  3+3j,
            (1,0,1,1) :  3+1j,
            (1,1,0,0) :  1-3j,
            (1,1,0,1) :  1-1j,
            (1,1,1,0) :  1+3j,
            (1,1,1,1) :  1+1j
        }
    else:
        sym_set = [1+1j, 1-1j, -1+1j, -1-1j]
        mapping_table = {
            (0,0) : 1+1j,
            (0,1) : 1-1j,
            (1,0) : -1+1j,
            (1,1) : -1-1j
        }
    if table_type == "mapping":
        return mapping_table
    elif table_type == "demapping":
        demapping_table = {v : k for k, v in mapping_table.items()}
        return demapping_table
    else:
        return sym_set

    
def mod(bits, mod_scheme):
    mapping_table = get_table(mod_scheme, "mapping")
    return np.array([mapping_table[tuple(b)] for b in bits])


# Mapping back to bits
def demod(xHat, mod_scheme):
    demapping_table = get_table(mod_scheme, "demapping")
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()])
    
    # calculate distance of each RX point to each possible point
    dists = abs(xHat.reshape((-1,1)) - constellation.reshape((1,-1)))
    
    # for each element in xHat, choose the index in constellation that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)
    
    # get back the real constellation point
    hardDecision = constellation[const_index]
    
    # transform the constellation point into the bit groups
    return np.vstack([demapping_table[C] for C in hardDecision])

def gen_data(B,Nt,Nr,snr_dB, mod_scheme, DELTA_SNR, correlated_model):
    
    sym_set = get_table(mod_scheme, "sym_set")
    snr_dB = np.random.uniform(low=snr_dB - DELTA_SNR, high=snr_dB + DELTA_SNR);
    SNR = 10.0 ** (snr_dB/10.0)
    if mod_scheme == "QPSK":
        Es = 2
    else:
        Es = 10
        
    sigma2 = Nt * Es / SNR # noise variance
    X = np.zeros([B,2*Nt])
    Y = np.zeros([B,2*Nr])
    Noise = np.zeros([B,2*Nr])
    Hb = np.zeros([B,2*Nr,2*Nt])
    HY = np.zeros([B,2*Nt])
    HH = np.zeros([B,2*Nt,2*Nt])
    for i in range(B):

        Hc = 1/np.sqrt(2)*(np.random.randn(Nr,Nt) + 1j*np.random.randn(Nr,Nt))
        if correlated_model == 1:
            R = one_ring_channel_model(Nt, 120) # compute covariance matrix R
            R_sqrt = scipy.linalg.sqrtm(R)
            Hc = R_sqrt * Hc
        H1 = np.concatenate((Hc.real, -Hc.imag), axis=1)
        H2 = np.concatenate((Hc.imag, Hc.real), axis=1)
        H = np.concatenate((H1, H2), axis=0)
        
        xc = np.random.choice(sym_set,Nt)
        x = np.concatenate((xc.real,xc.imag), axis=0)
        
        noise_c = 1/np.sqrt(2)*(np.random.randn(Nr) + 1j*np.random.randn(Nr))
        noise = np.concatenate((noise_c.real,noise_c.imag), axis=0)
        
        y = H.dot(x) + np.sqrt(sigma2)*noise
        
        X[i,:] = x
        HY[i,:] = H.T.dot(y)
        HH[i,:,:] = H.T.dot(H)
        Y[i,:] = y
        Noise[i,:] = noise
        Hb[i,:,:] = H
    return HY, HH, X, Y, Hb, Noise

def gen_data_QR(B,Nt,Nr,snr_dB, mod_scheme, DELTA_SNR, correlated_model):
    
    sym_set = get_table(mod_scheme, "sym_set")
    snr_dB = np.random.uniform(low=snr_dB - DELTA_SNR, high=snr_dB + DELTA_SNR);
    SNR = 10.0 ** (snr_dB/10.0)
    if mod_scheme == "QPSK":
        Es = 2
    else:
        Es = 10
        
    sigma2 = Nt * Es / SNR # noise variance
    X = np.zeros([B,2*Nt])
    Y = np.zeros([B,2*Nr])
    Noise = np.zeros([B,2*Nr])
    Hb = np.zeros([B,2*Nr,2*Nt])
    HY = np.zeros([B,2*Nt])
    RR = np.zeros([B,2*Nt,2*Nt])
    
    for i in range(B):

        Hc = 1/np.sqrt(2)*(np.random.randn(Nr,Nt) + 1j*np.random.randn(Nr,Nt))
        H1 = np.concatenate((Hc.real, -Hc.imag), axis=1)
        H2 = np.concatenate((Hc.imag, Hc.real), axis=1)
        H = np.concatenate((H1, H2), axis=0)
        Q, R = scipy.linalg.qr(H)
        
        xc = np.random.choice(sym_set,Nt)
        x = np.concatenate((xc.real,xc.imag), axis=0)
        
        noise_c = 1/np.sqrt(2)*(np.random.randn(Nr) + 1j*np.random.randn(Nr))
        noise = np.concatenate((noise_c.real,noise_c.imag), axis=0)
        
        y = H.dot(x) + np.sqrt(sigma2)*noise
        
        X[i,:] = x
        HY[i,:] = H.T.dot(y)
        RR[i,:,:] = R.T.dot(R)
        Y[i,:] = y
        Noise[i,:] = noise
        Hb[i,:,:] = H
    return HY, RR, X, Y, Hb, Noise

def gen_data_ZF(B,Nt,Nr,snr_dB, mod_scheme, DELTA_SNR, QR):
    
    sym_set = get_table(mod_scheme, "sym_set")
    snr_dB = np.random.uniform(low=snr_dB - DELTA_SNR, high=snr_dB + DELTA_SNR);
    SNR = 10.0 ** (snr_dB/10.0)
    if mod_scheme == "QPSK":
        Es = 2
    else:
        Es = 10
        
    sigma2 = Nt * Es / SNR # noise variance
    X = np.zeros([B,2*Nt])
    Y = np.zeros([B,2*Nr])
    Noise = np.zeros([B,2*Nr])
    Hb = np.zeros([B,2*Nr,2*Nt])
    HY = np.zeros([B,2*Nt])
    HH = np.zeros([B,2*Nt,2*Nt])
    X_ZF = np.zeros([B,2*Nt])
    X_0 = np.zeros([B,2*Nt])
    for i in range(B):

        Hc = 1/np.sqrt(2)*(np.random.randn(Nr,Nt) + 1j*np.random.randn(Nr,Nt))
        H1 = np.concatenate((Hc.real, -Hc.imag), axis=1)
        H2 = np.concatenate((Hc.imag, Hc.real), axis=1)
        H = np.concatenate((H1, H2), axis=0)
        Q, R = scipy.linalg.qr(H)
        
        xc = np.random.choice(sym_set,Nt)
        x = np.concatenate((xc.real,xc.imag), axis=0)
        
        noise_c = 1/np.sqrt(2)*(np.random.randn(Nr) + 1j*np.random.randn(Nr))
        noise = np.concatenate((noise_c.real,noise_c.imag), axis=0)
        
        y = H.dot(x) + np.sqrt(sigma2)*noise
        

        W = np.linalg.inv(H.T.dot(H)).dot(H.T)
        x_ZF = W.dot(y)
        x_ZF = np.zeros([1,2*Nt])
        X_ZF[i,:] = x_ZF
        
        X[i,:] = x
        HY[i,:] = H.T.dot(y)
        if QR == 0:
            HH[i,:,:] = H.T.dot(H)
        else:
            HH[i,:,:] = R.T.dot(R)
            
        Y[i,:] = y
        Noise[i,:] = noise
        Hb[i,:,:] = H
        
        xc_0 = np.random.choice(sym_set,Nt)
        x_0 = np.concatenate((xc_0.real,xc_0.imag), axis=0)
        X_0[i,:] = x_0
    return HY, HH, X, Y, Hb, Noise, X_ZF, X_0

import scipy
from scipy.integrate import quad

def compute_integral(func, theta, distance, Delta, D):
    
    def real_func(x):
        return scipy.real(func(x))
    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = quad(real_func, -Delta, Delta)
    imag_integral = quad(imag_func, -Delta, Delta)
    return (real_integral[0] + 1j*imag_integral[0])

def one_ring_channel_model(Nt, angular_spread):
    Delta = angular_spread*np.pi/180.0
    D = 1/2 # half wavelenth distance
    theta = np.pi/6 # angle of arrivale
    
    first_row = np.zeros([Nt,1]) + 1j*np.zeros([Nt,1])
    for col in range(Nt):
        distance = col # distance from the first antenna
        first_row[col,:] = compute_integral(lambda x: (np.exp(-1j*2*np.pi*D*distance*np.sin(x + theta))/(2*Delta)), theta, distance, Delta, D)
        
    # compute covariance matrix R
    R = scipy.linalg.toeplitz(first_row)
    return R
    
def get_symbol(x_hat, x, Nt, mod_scheme):
    N = 2*Nt
    x_hat_c = x_hat[:, 0:Nt] + 1j*x_hat[:, Nt:N]
    demapping_table = get_table(mod_scheme, "demapping")
    constellation = np.array([x for x in demapping_table.keys()])
    dists = abs(x_hat_c.reshape((-1,1)) - constellation.reshape((1,-1)))
    const_index = dists.argmin(axis=1)
    symbol = constellation[const_index]
    return symbol
    
def compute_BER(x_hat, x, Nt, mod_scheme):
    N = 2*Nt
    x_hat_c = x_hat[:, 0:Nt] + 1j*x_hat[:, Nt:N]
    x_c = x[:, 0:Nt] + 1j*x[:, Nt:N]
    x_hat_bin = demod(x_hat_c, mod_scheme)
    x_bin = demod(x_c, mod_scheme)
    BER = np.sum(abs(x_bin - x_hat_bin))/np.size(x_bin)
    return BER

def compute_BER_symbol(x_hat, x, Nt, mod_scheme):
    N = 2*Nt
    x_c = x[:, 0:Nt] + 1j*x[:, Nt:N]
    x_hat_bin = demod(np.array(x_hat), mod_scheme)
    x_bin = demod(x_c, mod_scheme)
    BER = np.sum(abs(x_bin - x_hat_bin))/np.size(x_bin)
    return BER

def compute_BER_vec(x_hat, x, Nt, mod_scheme):
    N = 2*Nt
    x_hat_c = x_hat[0:Nt] + 1j*x_hat[Nt:N]
    x_c = x[0:Nt] + 1j*x[Nt:N]
    x_hat_bin = demod(x_hat_c, mod_scheme)
    x_bin = demod(x_c, mod_scheme)
    BER = np.sum(abs(x_bin - x_hat_bin))/np.size(x_bin)
    return BER

def load_data(directory, SNR, j, n_train):
    mat = spio.loadmat(directory + str(SNR) + "dB.mat", squeeze_me=True)
    mat_2 = spio.loadmat(directory + str(SNR) + "dB_" + str(n_train) +  ".mat", squeeze_me=True)
    
    batch_X = mat['x']
    batch_X_ZF = mat['x_ZF']
    batch_X_hat = mat['x_hat']
    batch_X_hat_2 = mat_2['x_hat']
    batch_H = mat['H']
    batch_Y = mat['y']

    s = batch_X[j,:]
    H = batch_H[j,:,:]
    y = batch_Y[j,:]
    s_hat = np.sign(batch_X_hat[j,:])
    s_hat_2 = np.sign(batch_X_hat_2[j,:])
    s_ZF = np.sign(batch_X_ZF[j,:])
    
    _Hy_tmp = H.T.dot(y)
    _HH_tmp = H.T.dot(H)
    
    # Convert to 3D data
    _s =  np.sign(s[np.newaxis, :])
    _Hy = _Hy_tmp[np.newaxis, :]
    _HH = _HH_tmp[np.newaxis, :, :]
    m_hat = np.linalg.norm(y - H.dot(s_hat), ord=2)**2
    m_hat_2 = np.linalg.norm(y - H.dot(s_hat_2), ord=2)**2
    m_ZF = np.linalg.norm(y - H.dot(s_ZF), ord=2)**2
    m_s = np.linalg.norm(y - H.dot(s), ord=2)**2
    
    return s_hat, m_hat, H, y, _s, _Hy, _HH, s_ZF, m_ZF, m_s, s_hat_2, m_hat_2
        
def find_best_nb(y, H, cand, N, tabu_list):
    m_best = np.Infinity
    x_best = np.zeros([N,])
    for nn in range(0,N):
        x = copy.deepcopy(cand)
        x[nn] = -x[nn]
        m_x = np.linalg.norm(y-H.dot(x), ord=2)**2
        
        if m_x < m_best and not(m_x in tabu_list):
            m_best = m_x
            x_best = x
    return x_best, m_best

def sortSecond(val): 
    return val[1]

def find_best_nb_order(y, H, cand, N, tabu_list, best_order):
    list_m = []
    list_x = []
    n_nb = 0
    for nn in range(0,N):
        x = copy.deepcopy(cand)
        x[nn] = -x[nn]
        m_x = np.linalg.norm(y-H.dot(x), ord=2)**2
        if not(m_x in tabu_list):
            n_nb += 1
            list_m.append(m_x)
            list_x.append(x)
    list_m_tmp = copy.deepcopy(list_m)
    list_m_tmp.sort()
    m_best = list_m_tmp[best_order-1]
    x_best = list_x[list_m.index(m_best)]
    return x_best, m_best, n_nb

def get_general_params(SNR_vec, directory_data, directory_model, use_NN, change_direction, Nt, Nr, ss):
    SNR_dB = SNR_vec[ss]
    print('SNR_dB = ', SNR_dB, 'dB')
    
    m_hat_file_name = directory_data + "m.mat"
    model_file_name = directory_model + str(SNR_dB) + "/my_model_final.ckpt"
    if use_NN == 0:
        ber_file_name = directory_data + "ber_TS.mat"
    elif change_direction == 0:
        ber_file_name = directory_data + "ber_TSNN.mat"
    else:
        ber_file_name = directory_data + "ber_TSNN_CD.mat"
    
    alpha = 1.2 #1.2791
    Es = 2
    snr = 10**(SNR_dB/10)
    sigma2 = Nt*Es / snr; # noise variance
    T = alpha*sigma2*Nr   # metric threshold for early stop
    
    return m_hat_file_name, model_file_name, ber_file_name, T

def tabu_search(y, H, N, n_iter_max, s_ZF, m_ZF):
      
      # set up parameters for TS
      count = 0           # for early termiation
      i = 1               # number of searching iteration
      cand = np.sign(s_ZF)
      #s_hat = s_ZF
      m_hat = m_ZF
      tabu_list = [m_hat]  # save the metrics of tabu-vectors
      list_cand = [cand]
      while i <= n_iter_max:
          i += 1
          
          # Find best neighbor of x_hat
          x_best, m_best = find_best_nb_order(y, H, cand, N, tabu_list, 1)
      
          # update the current candidate
          cand = x_best
          #time_input = y - H.dot(cand)
          list_cand.append(cand)
          
          # update tabu list
          if len(tabu_list) > n_iter_max/2:
              tabu_list.pop(0)
          tabu_list.append(m_best)
          
          # update x_hat if better vector is found
          if m_best < m_hat:
              #s_hat = x_best
              m_hat = m_best
              count = 0
          else:
              count += 1
              
      return list_cand
 
def gen_data_RNN(B, Nt, Nr, snr_dB, mod_scheme, n_step, n_input):
    sym_set = get_table(mod_scheme, "sym_set")
    SNR = 10.0 ** (snr_dB/10.0)
    if mod_scheme == "QPSK":
        Es = 2
    else:
        Es = 10
        
    sigma2 = Nt * Es / SNR # noise variance
    dat_cand = np.zeros([B, n_step, n_input])
    dat_target = np.zeros([B, n_input])

    for i in range(B):
        Hc = 1/np.sqrt(2)*(np.random.randn(Nr,Nt) + 1j*np.random.randn(Nr,Nt))
        H1 = np.concatenate((Hc.real, -Hc.imag), axis=1)
        H2 = np.concatenate((Hc.imag, Hc.real), axis=1)
        H = np.concatenate((H1, H2), axis=0)
        
        xc = np.random.choice(sym_set,Nt)
        x = np.concatenate((xc.real,xc.imag), axis=0)
        
        wc = 1/np.sqrt(2)*(np.random.randn(Nr) + 1j*np.random.randn(Nr))
        w = np.concatenate((wc.real,wc.imag), axis=0)
        
        y = H.dot(x) + np.sqrt(sigma2)*w
        W = np.linalg.inv(H)
        x_ZF = W.dot(y)
        m_ZF = np.linalg.norm(y - H.dot(x_ZF), ord=2)**2
        dat_target[i,:] = x
        
        # get list of moves by TS algorithm
        n_iter_max = 19
        list_cand = tabu_search(y, H, 2*Nt, n_iter_max, x_ZF, m_ZF)
        dat_cand[i, :, :] = list_cand
        dat_target[i, :] = x
    return dat_cand, dat_target
                           
if __name__ == '__main__':
    B = 2
    Nr = 8
    Nt = 8
    mod_scheme = "QPSK"
    snr_dB = 10
    SNR = 10.0 ** (snr_dB/10.0)
    Es = 2
    sigma2 = Nt * Es / SNR # noise variance
    batch_HY, batch_HH, batch_X,_,_ = gen_data(B,Nt,Nr,snr_dB, mod_scheme, 0)

