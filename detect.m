function detect(Nr, Nt, snr_dB, m_scheme, detector, n_iter, ss, data_filename, L, K)

global sim_ber sim_com sim_node

N = 2*Nt; M = 2*Nr; snr = 10^(snr_dB/10);
if strcmp(m_scheme,'QPSK')
    symset = [-1,1];        m_order = 4;    Es = 2;
else
    symset = [-3,-1,1,3];   m_order = 16;   Es = 10;
end
n_bit = log2(m_order);

%% =========== Load NNs and complexity of NNs =============================
J = 4; DL_IR_SD_net = []; DPP_SD_net = []; n_oper_DL_IR = 0; n_oper_DPP = 0;
file_name = strcat('.\model\',num2str(Nt),'x',num2str(Nr),'_Ref\');

if detector(4) == 1
    % DL-IR-SD
    net1 = load(strcat(file_name,'DL_IR_SD'));
    DL_IR_SD_net = net1.NN_path;
    n_oper_DL_IR = 2*(544*128 + 128*J);
end

if detector(5)==1
    % DPP-SD
    net2 = load(strcat(file_name,'DPP_SD'));
    DPP_SD_net = net2.NN_path;
    nSymbols = 4;
    nHidden = 2*(Nt + nSymbols);
    
    % Neural Network Complexity
    NN_OPER = nHidden*(2*Nt+2)+nSymbols*nHidden;
    % Complexity for generating input
    INPUT_MULT_1 = 2*Nt;                        % z'z
    INPUT_ADD_1  = 2*Nt - 1;                    % z'z
    INPUT_MULT_2 = 2*Nt+4*sum(1:Nt-1);          % R'z
    INPUT_ADD_2  = 2*(Nt-1)+4*sum(1:Nt-1);      % R'z
    INPUT_MULT   = INPUT_MULT_1 + INPUT_MULT_2;
    INPUT_ADD   = INPUT_ADD_1 + INPUT_ADD_2;
    n_oper_DPP = NN_OPER + INPUT_MULT + INPUT_ADD;
end
% FS-Net
n_oper_FSNet = (2*M - 1)*N^2 + (2*M - 1)*N + L*(2*N^2 + 4*N);
%% ============================================================

ber_SD_FP = zeros(n_iter,1);    com_SD_FP = zeros(n_iter,1);    node_SD_FP = zeros(n_iter,1);
ber_SD_SE = zeros(n_iter,1);    com_SD_SE = zeros(n_iter,1);    node_SD_SE = zeros(n_iter,1);
ber_SD_DL = zeros(n_iter,1);    com_SD_DL = zeros(n_iter,1);    node_SD_DL = zeros(n_iter,1);
ber_SD_DL_IR = zeros(n_iter,1); com_SD_DL_IR = zeros(n_iter,1); node_SD_DL_IR = zeros(n_iter,1);
ber_SD_DPP = zeros(n_iter,1);   com_SD_DPP = zeros(n_iter,1);   node_SD_DPP = zeros(n_iter,1);
ber_KSD = zeros(n_iter,1);      com_KSD = zeros(n_iter,1);      node_KSD = zeros(n_iter,1);
ber_KSD_DL = zeros(n_iter,1);   com_KSD_DL = zeros(n_iter,1);   node_KSD_DL = zeros(n_iter,1);
ber_KSD_DL_noOrder = zeros(n_iter,1);   com_KSD_DL_noOrder = zeros(n_iter,1);   node_KSD_DL_noOrder = zeros(n_iter,1);

%% Load data
data = load(data_filename);
batch_S_NN = data.s_nn;
batch_S = data.s;
batch_Hb = data.H;
batch_Y = data.y;
batch_N = data.n;

parfor nn = 1:n_iter
    
    sigma2 = Nt*Es / snr; % noise variance
    
    s = batch_S(nn,:).';
    y = batch_Y(nn,:).';
    n = batch_N(nn,:).';
    Htmp = batch_Hb(nn,:,:);
    H = reshape(Htmp,2*Nr,2*Nt);
    s_c = s(1:Nt) + 1i*s(Nt+1:end);
    s_int = qamdemod(s_c,m_order);
    s_bin = de2bi(s_int, n_bit);
    
    alpha = gammaincinv(0.99,Nr,'lower')/Nr;
    radius = alpha * 2*Nr * sigma2;
    
    if detector(1) % FP-SD
        [x_hat, n_oper, n_node] = SD_FP(H, y, radius, symset);
        x_hat_c = x_hat(1:Nt) + 1i*x_hat(Nt+1:end);
        x_hat_demod = qamdemod(x_hat_c, m_order);
        x_hat_bin = de2bi(x_hat_demod, n_bit);
        ber_SD_FP(nn) = biterr(s_bin, x_hat_bin);
        com_SD_FP(nn) = n_oper;
        node_SD_FP(nn) = n_node;
    end
    
    if detector(2) % SE-SD
        [x_hat, n_oper, n_node] = SD_SE(H, y, radius, symset);
        x_hat_c = x_hat(1:Nt) + 1i*x_hat(Nt+1:end);
        x_hat_demod = qamdemod(x_hat_c, m_order);
        x_hat_bin = de2bi(x_hat_demod, n_bit);
        ber_SD_SE(nn) = biterr(s_bin, x_hat_bin);
        com_SD_SE(nn) = n_oper;
        node_SD_SE(nn) = n_node;
    end
    
    if detector(3) % DL-SD
        s_NN_raw = squeeze(batch_S_NN(nn,:,:)).';
        s_NN = mod_slicer(s_NN_raw, m_scheme).';
        radius = Inf;%norm(y-H*s_NN)^2;
        % ordering ==============
        [y1, H1, s_NN_raw1, s_NN1, s_bin1, s1] = ...
            order_Tx(y,H,n,s,s_bin,s_NN_raw,s_NN,m_order,n_bit,sigma2,1);
        %=============================
        [x_hat, n_oper, n_node] = SD_DL(H1, y1, radius, symset, s_NN_raw1);
        x_hat_c = x_hat(1:Nt) + 1i*x_hat(Nt+1:end);
        x_hat_demod = qamdemod(x_hat_c, m_order);
        x_hat_bin = de2bi(x_hat_demod, n_bit);
        ber_SD_DL(nn) = biterr(s_bin1, x_hat_bin);
        com_SD_DL(nn) = n_oper + n_oper_FSNet;
        node_SD_DL(nn) = n_node;
    end
    
    if detector(4) % DL-SD, radius estimation
        %[x_hat, n_oper, n_node] = SD_DL_IR(y, H, symset, snr, J, DL_IR_SD_net);
        [x_hat, n_oper, n_node] = SD_DL_IR(y, H, symset, snr, J, DPP_SD_net);
        x_hat_c = x_hat(1:Nt) + 1i*x_hat(Nt+1:end);
        x_hat_demod = qamdemod(x_hat_c, m_order);
        x_hat_bin = de2bi(x_hat_demod, n_bit);
        ber_SD_DL_IR(nn) = biterr(s_bin, x_hat_bin);
        com_SD_DL_IR(nn) = n_oper + n_oper_DL_IR;
        node_SD_DL_IR(nn) = n_node;
    end
    
    if detector(5) % DPP-SD, Doyeon's improved radius estimmation
        [x_hat, n_oper, n_node] = SD_DPP(H, y, radius, sigma2, DPP_SD_net, snr_dB);
        x_hat_c = x_hat;%(1:Nt) + 1i*x_hat(Nt+1:end);
        x_hat_demod = qamdemod(x_hat_c, m_order);
        x_hat_bin = de2bi(x_hat_demod, n_bit);
        ber_SD_DPP(nn) = biterr(s_bin, x_hat_bin);
        com_SD_DPP(nn) = n_oper + n_oper_DPP;
        node_SD_DPP(nn) = n_node;
    end
    
    if detector(6) % KSD
        s_NN_raw = squeeze(batch_S_NN(nn,:,:)).';
        s_NN = mod_slicer(s_NN_raw, m_scheme).';
        [x_hat, n_oper, n_node] = KSD(H, y, symset, K);
        x_hat_c = x_hat(1:Nt) + 1i*x_hat(Nt+1:end);
        x_hat_demod = qamdemod(x_hat_c, m_order);
        x_hat_bin = de2bi(x_hat_demod, n_bit);
        ber_KSD(nn) = biterr(s_bin, x_hat_bin);
        com_KSD(nn) = n_oper;
        node_KSD(nn) = n_node;
    end
    
    if detector(7) % KSD-ML
        s_NN_raw = squeeze(batch_S_NN(nn,:,:)).';
        s_NN = mod_slicer(s_NN_raw, m_scheme).';
        % ordering ==============
        [y1, H1, s_NN_raw1, s_NN1, s_bin1, s1] = ...
            order_Tx(y,H,n,s,s_bin,s_NN_raw,s_NN,m_order,n_bit,sigma2,1);
        %=============================
        [x_hat, n_oper, n_node] = KSD_DL(H1, y1, symset, s_NN1, K);
        x_hat_c = x_hat(1:Nt) + 1i*x_hat(Nt+1:end);
        x_hat_demod = qamdemod(x_hat_c, m_order);
        x_hat_bin = de2bi(x_hat_demod, n_bit);
        ber_KSD_DL(nn) = biterr(s_bin1, x_hat_bin);
        com_KSD_DL(nn) = n_oper + n_oper_FSNet;
        node_KSD_DL(nn) = n_node;
    end
    
    if detector(8) % KSD-ML, no order
        s_NN_raw = squeeze(batch_S_NN(nn,:,:)).';
        s_NN = mod_slicer(s_NN_raw, m_scheme).';
        % ordering ==============
        [y1, H1, s_NN_raw1, s_NN1, s_bin1, s1] = ...
            order_Tx(y,H,n,s,s_bin,s_NN_raw,s_NN,m_order,n_bit,sigma2,0);
        %=============================
        [x_hat, n_oper, n_node] = KSD_DL(H1, y1, symset, s_NN1, K);
        x_hat_c = x_hat(1:Nt) + 1i*x_hat(Nt+1:end);
        x_hat_demod = qamdemod(x_hat_c, m_order);
        x_hat_bin = de2bi(x_hat_demod, n_bit);
        ber_KSD_DL_noOrder(nn) = biterr(s_bin1, x_hat_bin);
        com_KSD_DL_noOrder(nn) = n_oper + n_oper_FSNet;
        node_KSD_DL_noOrder(nn) = n_node;
    end
end

% tmp
sim_ber(ss, :) = [mean(ber_SD_FP), mean(ber_SD_SE), mean(ber_SD_DL), mean(ber_SD_DL_IR), mean(ber_SD_DPP), mean(ber_KSD), mean(ber_KSD_DL), mean(ber_KSD_DL_noOrder)]/(n_bit*Nt);
sim_com(ss, :) = [mean(com_SD_FP), mean(com_SD_SE), mean(com_SD_DL), mean(com_SD_DL_IR), mean(com_SD_DPP), mean(com_KSD), mean(com_KSD_DL), mean(com_KSD_DL_noOrder)];
sim_node(ss, :) = [mean(node_SD_FP), mean(node_SD_SE), mean(node_SD_DL), mean(node_SD_DL_IR), mean(node_SD_DPP), mean(node_KSD), mean(node_KSD_DL), mean(node_KSD_DL_noOrder)];

end % eof