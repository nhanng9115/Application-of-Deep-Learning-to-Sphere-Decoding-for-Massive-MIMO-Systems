function [x_hat, nOper, nNode] = sphec(H, y, radius, m_scheme)
n = size(H,2);

if strcmp(m_scheme,'QPSK')
    symset = [-1,1];
else
    symset = [-3,-1,1,3];
end

global RADIUS X_HAT X_TMP SYMSET N_MUL N_ADD N_NODE;
RADIUS = radius;
X_HAT = zeros(n, 1);
X_TMP = zeros(n, 1);
SYMSET = symset;
N_MUL  = 0; N_ADD  = 0; N_NODE = 0;

[Q, R] = qr(H, 0);
N_MUL = N_MUL + 2/3*n^3 + n^2 + n/3 - 2;
N_ADD = N_ADD + 2/3*n^3 + 1/2*n^2 + 11/6 - 3;

z = Q'*y;
N_MUL = N_MUL + n^2;
N_ADD = N_ADD + n*(n-1);

sphdec_core(z, R, n, 0);

x_hat = X_HAT;
nOper = N_MUL + N_ADD;
nNode = N_NODE;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function sphdec_core(z, R, layer, dist)
n = size(R,2);
global RADIUS X_HAT X_TMP SYMSET N_MUL N_ADD N_NODE;

for ii = 1:length(SYMSET)
    X_TMP(layer) = SYMSET(ii);
    d = abs(z(layer) - R(layer,[layer:end])*X_TMP(layer:end))^2 + dist;
    
    % count
    n_layer = n - layer + 1;
    N_MUL = N_MUL + n_layer;
    N_ADD = N_ADD + (n_layer+1);
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


