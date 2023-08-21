function [x_hat, nOper, nNode] = KSD(H, y, symset, K)
n = size(H,2);

global N_MUL N_ADD N_NODE
N_MUL  = 1; N_ADD  = 1; N_NODE = 1;

[Q,R] = qr(H);
N_MUL = N_MUL + 2/3*n^3 + n^2 + n/3 - 2;
N_ADD = N_ADD + 2/3*n^3 + 1/2*n^2 + 11/6*n - 3;

z = Q'*y;
N_MUL = N_MUL + n^2;
N_ADD = N_ADD + n*(n-1);

x_layer_sorted = [];

for layer = n:-1:1
    
    d_list = [];
    x_list = [];
    n_element = n - layer + 1;
    
    if layer == n
        for i = 1:length(symset)
            d = abs(z(layer) - R(layer,layer:end)*symset(i))^2;
            x_list = cat(2,x_list,symset(i));
            d_list = cat(2,d_list,d);
        end
    else
        K_layer = min(K,size(x_layer_sorted,2));
        for k = 1:K_layer
            x_upper_layer = x_layer_sorted(:,k);
            
            for i = 1:length(symset)
                x_tmp = [symset(i); x_upper_layer];
                d = d_layer_sorted(k) + abs(z(layer) - R(layer,layer:end)*x_tmp)^2;
                
                x_list = cat(2,x_list,x_tmp);
                d_list = cat(2,d_list,d);
                
                % count
                N_MUL = N_MUL + n_element + 1;
                N_ADD = N_ADD + (n_element+1);
                N_NODE = N_NODE + 1;
            end
        end
    end
    [d_layer_sorted, idx] = sort(d_list);
    x_layer_sorted = x_list(:,idx);
end
x_hat = x_layer_sorted(:,1);

nOper = N_MUL + N_ADD;
nNode = N_NODE;
end