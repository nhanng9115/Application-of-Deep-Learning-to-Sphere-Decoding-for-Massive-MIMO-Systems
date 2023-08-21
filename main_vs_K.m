clear all
clc;
close all
warning off
% *****************************
% m_scheme = 'QPSK';
m_scheme = '16QAM';

K_vec = [32,64,128,256];
if strcmp(m_scheme,'QPSK')
    Nt = 32; Nr = Nt; L = 15;
else
    Nt = 16; Nr = Nt; L = 30;
end
snr_dB = 20;
n_iter = 7000;
detector = [0,... %conv. FP-SD
    0,... %conv. SE-SD
    0,... %FDL-SD
    0,... %MR-RD-SD
    0,... %DPP-SD
    1,... %conv. KSD
    1,... %FDL-KSD, order
    1];% FDL-KSD, no order
line_style = {'-ko', '-bs', '-r*', '--k+', ':g^', '-bo', '--r*', ':md'};

global sim_ber sim_com sim_node
sim_ber = zeros(length(K_vec), length(detector));
sim_com = zeros(length(K_vec), length(detector));
sim_node = zeros(length(K_vec), length(detector));

% To load testing data
file_name = strcat('.\model\',num2str(Nt),'x',num2str(Nr),'_',m_scheme,'\');

for ss = 1:length(K_vec)
    K = K_vec(ss)
    data_filename = strcat(file_name,num2str(snr_dB),'dB.mat');
    detect(Nr, Nt, snr_dB, m_scheme, detector, n_iter, ss, data_filename, L, K);
end
% sim_ber
% sim_com

% plot_fig(K_vec, detector, m_scheme, Nt, Nr, 1, 1, 0, line_style)
plot_fig_K(detector, m_scheme, Nt, Nr, line_style)

