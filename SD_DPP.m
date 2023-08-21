function [xhat,n_oper,n_node] = SD_DPP(Hr, yr, radius_conv, sigma2, DPP_SD_net, snr_dB)

global N_MUL N_ADD N_NODE;
N_MUL  = 0; N_ADD  = 0; N_NODE = 0;

% convert real model back to complex model
Nt = size(Hr,2)/2;
H = Hr(1:Nt,1:Nt) + 1j*Hr(Nt+1:end,1:Nt);
y = yr(1:Nt) + 1i*yr(Nt+1:end);
symset = loadAlphabet('QPSK');

[Q, R] = qr(H, 0);
z = Q'*y;
% Complexity for QR decomposition and compute z
N_MUL = N_MUL + 2/3*Nt^3 + Nt^2 + Nt/3 - 2;
N_ADD = N_ADD + 2/3*Nt^3 + 1/2*Nt^2 + 11/6*Nt - 3;
N_MUL = N_MUL + Nt^2;
N_ADD = N_ADD + Nt*(Nt-1);

%% using DNN to estimate radius ===============================
Rz = R'*z; 
Rz_data = [real(Rz); imag(Rz) ];
input_Data = [z'*z; Rz_data; sigma2];
% N_MUL = N_MUL + Nt^2 + 2*Nt;

ES_P = DPP_SD_net(input_Data);
[ES_P, node_I] = sort(ES_P);
[m1,m2] = tradeoff_opt_SNR(snr_dB);
radius = min(sqrt(radius_conv),m1*ES_P(1));
%%==============================================================

% N_ADD examine this variable before make it global
global RADIUS X_HAT X_TMP SYMBSETSIZE SEARCHFLAG SYMSET;
global node_decision estimated_distance ET_C M_C;

RADIUS = radius^2;            % Initial Radius
n = size(R,2);                       % number of layers

X_HAT        = zeros(n, 1);
X_TMP        = zeros(n, 1);
SYMBSETSIZE   = length(symset(:));
SEARCHFLAG    = 0;
ET_C          = 0;
node_decision = node_I;
estimated_distance = ES_P;
M_C           = m2;
SYMSET = symset;

% starting tree search
sphdec_core(z, R, symset, n, 0);

if SEARCHFLAG > 0
    xhat = X_HAT;
else
    W_ZF = (R'*R)^(-1) * R';
    xhat = W_ZF*z;
    N_MUL = N_MUL + n^2;
end

n_oper = 3*N_MUL + 2*N_ADD;
n_node = N_NODE;
clear RADIUS X_HAT SYMBSETSIZE SEARCHFLAG;
end

function sphdec_core(z, R, symset, layer, dist)

n = size(R,2);
global RADIUS X_HAT X_TMP SYMBSETSIZE SEARCHFLAG SYMSET;
global node_decision estimated_distance ET_C M_C;
global N_MUL N_ADD N_NODE;

if layer == n
    for pp = 1:SYMBSETSIZE
        X_TMP(end) = symset(node_decision(pp));
        B = abs(z(end)-R(end,end)*X_TMP(end))^2;
        N_MUL  = N_MUL + 3;
        N_ADD   = N_ADD  + 2;
        if ( B < RADIUS )
            sphdec_core(z, R, symset, layer-1, B);
        end
        if (SEARCHFLAG > 0 && pp < SYMBSETSIZE)
            N_MUL = N_MUL + 1;
            N_ADD  = N_ADD + 1;
            if ( M_C*sqrt(RADIUS) < estimated_distance(pp+1) )
                ET_C = 1;
                break;             % == Early termination == %
            end
        end
    end
else
    % ordering based on SE-SD
    symset_dis = abs(z(layer) - R(layer,layer)*SYMSET);
    N_MUL = N_MUL + length(SYMSET);
    N_ADD = N_ADD + length(SYMSET);
    [~, i_sort] = sort(symset_dis, 'ascend');
    symset_order = SYMSET(i_sort);
    
    z_higher_layer = z(layer) - R(layer,[layer+1:end])*X_TMP(layer+1:end);
    % count
    n_layer_higher = n - layer;
    N_MUL = N_MUL + n_layer_higher;
    N_ADD = N_ADD + (n_layer_higher+1);
    
    for ii = 1:length(symset_order)
        X_TMP(layer) = symset_order(ii);
        d = abs(z_higher_layer - R(layer,layer)*X_TMP(layer))^2 + dist;
        % count
        N_MUL = N_MUL + 2;
        N_ADD = N_ADD + 2;
        N_NODE = N_NODE + 1;
        
        if (d <= RADIUS)
            if layer == 1
                X_HAT =  X_TMP;
                RADIUS =  d;
                SEARCHFLAG = 1;
            else
                sphdec_core(z, R, symset, layer-1, d);
            end
        end
    end
end
end
% function [xhat,n_oper,n_node] = SD_DPP(Hr, yr, radius_conv, sigma2, DPP_SD_net, snr_dB)
% 
% % convert real model back to complex model
% Nt = size(Hr,2)/2;
% H = Hr(1:Nt,1:Nt) + 1j*Hr(Nt+1:end,1:Nt);
% y = yr(1:Nt) + 1i*yr(Nt+1:end);
% symset = loadAlphabet('QPSK');
% 
% [Q, R] = qr(H, 0);
% z = Q'*y;
% %% using DNN to estimate radius ===============================
% Rz = R'*z; Rz_data = [real(Rz); imag(Rz) ];
% input_Data = [z'*z; Rz_data; sigma2];
% 
% ES_P = DPP_SD_net(input_Data);
% [ES_P, node_I] = sort(ES_P);
% [m1,m2] = tradeoff_opt_SNR(snr_dB);
% radius = min(sqrt(radius_conv),m1*ES_P(1));
% %%==============================================================
% 
% % N_ADD examine this variable before make it global
% global RADIUS X_HAT X_TMP SYMBSETSIZE SEARCHFLAG SYMSET;
% global node_decision estimated_distance ET_C M_C;
% global N_MUL N_ADD N_NODE;
% 
% RADIUS = radius^2;            % Initial Radius
% n = size(R,2);                       % number of layers
% 
% X_HAT        = zeros(n, 1);
% X_TMP        = zeros(n, 1);
% SYMBSETSIZE   = length(symset(:));
% SEARCHFLAG    = 0;
% ET_C          = 0;
% node_decision = node_I;
% estimated_distance = ES_P;
% M_C           = m2;
% SYMSET = symset;
% N_MUL  = 0; N_ADD  = 0; N_NODE = 0;
% 
% 
% % Complexity for QR decomposition and compute z
% % N_MUL = N_MUL + 2/3*n^3 + n^2 + n/3 - 2;
% % N_ADD = N_ADD + 2/3*n^3 + 1/2*n^2 + 11/6*n - 3;
% % N_MUL = N_MUL + n^2;
% % N_ADD = N_ADD + n*(n-1);
% 
% 
% % starting tree search
% sphdec_core(z, R, symset, n, 0);
% 
% if SEARCHFLAG > 0
%     xhat = X_HAT;
% else
%     W_ZF = (R'*R)^(-1) * R';
%     xhat = W_ZF*z;
%     N_MUL = N_MUL + n^2;
% end
% 
% n_oper = 2*(N_MUL + N_ADD);
% n_node = N_NODE;
% clear RADIUS X_HAT SYMBSETSIZE SEARCHFLAG;
% end
% 
% function sphdec_core(z, R, symset, layer, dist)
% 
% n = size(R,2);
% global RADIUS X_HAT X_TMP SYMBSETSIZE SEARCHFLAG SYMSET;
% global node_decision estimated_distance ET_C M_C;
% global N_MUL N_ADD N_NODE;
% 
% if layer == n
%     for pp = 1:SYMBSETSIZE
%         X_TMP(end) = symset(node_decision(pp));
%         B = abs(z(end)-R(end,end)*X_TMP(end))^2;
%         N_MUL  = N_MUL + 3;
%         N_ADD   = N_ADD  + 2;
%         if ( B < RADIUS )
%             sphdec_core(z, R, symset, layer-1, B);
%         end
%         if (SEARCHFLAG > 0 && pp < SYMBSETSIZE)
%             N_MUL = N_MUL + 1;
%             N_ADD  = N_ADD + 1;
%             if ( M_C*sqrt(RADIUS) < estimated_distance(pp+1) )
%                 ET_C = 1;
%                 break;             % == Early termination == %
%             end
%         end
%     end
% else
%     % ordering based on SE-SD
%     symset_dis = abs(z(layer) - R(layer,layer)*SYMSET);
%     N_MUL = N_MUL + length(SYMSET);
%     N_ADD = N_ADD + length(SYMSET);
%     [~, i_sort] = sort(symset_dis, 'ascend');
%     symset_order = SYMSET(i_sort);
%     
%     z_higher_layer = z(layer) - R(layer,[layer+1:end])*X_TMP(layer+1:end);
%     % count
%     n_layer_higher = n - layer;
%     N_MUL = N_MUL + n_layer_higher;
%     N_ADD = N_ADD + (n_layer_higher+1);
%     
%     for ii = 1:length(symset_order)
%         X_TMP(layer) = symset_order(ii);
%         d = abs(z_higher_layer - R(layer,layer)*X_TMP(layer))^2 + dist;
%         % count
%         N_MUL = N_MUL + 2;
%         N_ADD = N_ADD + 2;
%         N_NODE = N_NODE + 1;
%         
%         if (d <= RADIUS)
%             if layer == 1
%                 X_HAT =  X_TMP;
%                 RADIUS =  d;
%                 SEARCHFLAG = 1;
%             else
%                 sphdec_core(z, R, symset, layer-1, d);
%             end
%         end
%     end
% end
% end