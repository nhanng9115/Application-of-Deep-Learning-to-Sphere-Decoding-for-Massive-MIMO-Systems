clear all
clc;
close all
warning off
% *****************************
% m_scheme = 'QPSK';
m_scheme = '16QAM';

if strcmp(m_scheme,'QPSK')
    Nt = 24; Nr = Nt; L = 12; K = 256;
    SNR_dB_vec = 4:2:12;
else
    Nt = 16; Nr = Nt; L = 30; K = 256;
    SNR_dB_vec = 12:2:20;
end

n_iter = 1000*[1:7];
detector = [0,... %conv. FP-SD
    0,... %conv. SE-SD
    0,... %FDL-SD
    0,... %MR-RD-SD
    0,... %DPP-SD
    1,... %conv. KSD
    1,... %FDL-KSD, order
    1];% FDL-KSD, no order
line_style = {'-ko', '-bs', '-r*', '--k+', ':g^', '-bo', '--r*', ':kd'};

global sim_ber sim_com sim_node
sim_ber = zeros(length(SNR_dB_vec), length(detector));
sim_com = zeros(length(SNR_dB_vec), length(detector));
sim_node = zeros(length(SNR_dB_vec), length(detector));
% To load testing data
% file_name = strcat('.\data\',m_scheme,'_',num2str(Nt),'\');
file_name = strcat('.\model\',num2str(Nt),'x',num2str(Nr),'_',m_scheme,'\');

for ss = 1:length(SNR_dB_vec)
    snr_dB = SNR_dB_vec(ss);
    data_filename = strcat(file_name,num2str(snr_dB),'dB.mat');
    disp(strcat('SNR = ', num2str(snr_dB), ' [dB]'));
    detect(Nr, Nt, snr_dB, m_scheme, detector, n_iter(ss), ss, data_filename, L, K);
end
sim_ber
sim_com
% sim_node
% reduce_comp = (sim_com(:,2) - sim_com(:,3))./sim_com(:,2) * 100
% reduce_node = (sim_node(:,2) - sim_node(:,3))./sim_node(:,2) * 100

plot_fig(SNR_dB_vec, detector, m_scheme, Nt, Nr, 1, 1, 0, line_style)
% hold all
