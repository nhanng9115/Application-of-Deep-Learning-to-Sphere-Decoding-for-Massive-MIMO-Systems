function [xhat, n_oper, n_node] = SD_DL_IR(y, H, symset, snr, J, DL_SD_net)
n = size(H,2);
global RADIUS X_HAT X_TMP SYMSET N_MUL N_ADD N_NODE SEARCHFLAG;
SEARCHFLAG    = 0;
X_HAT = zeros(n, 1);
X_TMP = zeros(n, 1);
SYMSET = symset;
N_MUL  = 0; N_ADD  = 0; N_NODE = 0;

[Q, R] = qr(H, 0);
N_MUL = N_MUL + 2/3*n^3 + n^2 + n/3 - 2;
N_ADD = N_ADD + 2/3*n^3 + 1/2*n^2 + 11/6*n - 3;

z = Q'*y;
N_MUL = N_MUL + n^2;
N_ADD = N_ADD + n*(n-1);

%% using DNN to estimate radius ===============================
Nt = n/2; Nr = n/2;
Hc = H(1:Nt,1:Nt) * 1j*H(Nt+1:end,1:Nt);
H_ = reshape(Hc,[Nt*Nr,1]);
H_data = [real(H_); imag(H_) ];
input_Data = [y; H_data];
radius_set = DL_SD_net(input_Data);
RADIUS = radius_set(1)^2;
%%==============================================================

% start tree search
sphdec_core(z, R, n, 0);

% conclusion
if SEARCHFLAG > 0
    xhat = X_HAT;
else
    for q = 2:J
        RADIUS = radius_set(q)^2;
        sphdec_core(z, R, n, 0);
        if SEARCHFLAG > 0
            xhat = X_HAT;
            break;
        elseif q == J && SEARCHFLAG == 0
            xhat = (R'*R+(n/snr)*eye(n))\(R'*z);
            N_MUL = N_MUL + (2*n - 1)*n^2/2 + n + n^3 + (n - 1)*n/2;
        end
    end
end
n_oper = N_MUL + N_ADD;
n_node = N_NODE;
clear SPHDEC_RADIUS RETVAL SYMBSETSIZE SEARCHFLAG;

function sphdec_core(z, R, layer, dist)
n = size(R,2);

global RADIUS X_HAT X_TMP SYMSET N_MUL N_ADD N_NODE SEARCHFLAG;

% ordering based on SE-SD
symset_dis = abs(z(layer) - R(layer,layer)*SYMSET);
N_MUL = N_MUL + length(SYMSET);
N_ADD = N_ADD + length(SYMSET);
[~, i_sort] = sort(symset_dis, 'ascend');
symset_order = SYMSET(i_sort);

% compute UB, LB
z_higher_layer = z(layer) - R(layer,[layer+1:end])*X_TMP(layer+1:end);
d_higher_layer = sqrt(RADIUS - dist);
bound_1 = floor((z_higher_layer + d_higher_layer)/R(layer,layer));
bound_2 = ceil((z_higher_layer - d_higher_layer)/R(layer,layer));
% count
n_layer_higher = n - layer;
N_MUL = N_MUL + n_layer_higher;
N_ADD = N_ADD + (n_layer_higher+1);

UB = min(max(bound_1,bound_2),SYMSET(end));
LB = max(min(bound_1,bound_2),SYMSET(1));

for ii = 1:length(symset_order)
    X_TMP(layer) = symset_order(ii);
    if X_TMP(layer) >= LB && X_TMP(layer) <= UB
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
                sphdec_core(z, R, layer-1, d);
            end
        end
    end
end

