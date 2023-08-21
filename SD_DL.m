function [x_hat, nOper, nNode] = SD_DL(H, y, radius, symset, s_NN_raw)
n = size(H,2);

global RADIUS X_HAT X_TMP SYMSET N_MUL N_ADD N_NODE S_NN_RAW
% S_NN = mod_slicer(s_NN_raw, m_scheme).';
X_HAT = zeros(n, 1);
X_TMP = zeros(n, 1);
SYMSET = symset;
S_NN_RAW = s_NN_raw;
N_MUL  = 0; N_ADD  = 0; N_NODE = 0;

[Q, R] = qr(H, 0);
N_MUL = N_MUL + 2/3*n^3 + n^2 + n/3 - 2;
N_ADD = N_ADD + 2/3*n^3 + 1/2*n^2 + 11/6*n - 3;

z = Q'*y;
N_MUL = N_MUL + n^2;
N_ADD = N_ADD + n*(n-1);

RADIUS = radius;
sphdec_core(z, R, n, 0);

% if X_HAT(1) == 0
%     x_hat = S_NN;
%     %disp("*****")
% else
%     x_hat = X_HAT;
% end

x_hat = X_HAT;

nOper = N_MUL + N_ADD;
nNode = N_NODE;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function sphdec_core(z, R, layer, dist)
n = size(R,2);

global RADIUS X_HAT X_TMP SYMSET N_MUL N_ADD S_NN_RAW N_NODE S_NN;

symset_dis = abs(SYMSET - S_NN_RAW(layer));
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
% S_NN_RAW(layer)
% 1;
for ii = 1:length(symset_order)
    X_TMP(layer) = symset_order(ii);
    if X_TMP(layer) >= LB && X_TMP(layer) <= UB% && X_TMP(layer)~=S_NN(layer)
        d = abs(z_higher_layer - R(layer,layer)*X_TMP(layer))^2 + dist;
        % count
        N_MUL = N_MUL + 2;
        N_ADD = N_ADD + 2;
        N_NODE = N_NODE + 1;
        
        if (d <= RADIUS)
            if layer == 1
                X_HAT =  X_TMP;
                RADIUS =  d;
            else
                sphdec_core(z, R, layer-1, d);
            end
        end
    end
end
end


