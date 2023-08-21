function [y1, H1, s_NN_raw1, s_NN1, s_bin1, s_1] = order_Tx(y,H,n,s,s_bin,s_NN_raw,s_NN,m_order,n_bit,sigma2,ordering)

if ordering == 1
    Nt = size(H,2)/2;
    [tmp, idx_asc] = sort(abs(s_NN_raw - s_NN),'descend');
    H1 = H(:,idx_asc);
    
    s_1 = s(idx_asc); % order Tx signal as well
    
    s_c_1 = s_1(1:Nt) + 1i*s_1(Nt+1:end);
    sMod_1 = qamdemod(s_c_1,m_order,'gray');
    s_bin1 = de2bi(sMod_1, n_bit);
    
    y1 = H1*s_1 + sqrt(sigma2)*n;
    
    s_NN_raw1 = s_NN_raw(idx_asc);
    s_NN1 = s_NN(idx_asc);
else
    y1 = y; H1 = H; s_NN_raw1 = s_NN_raw; s_NN1 = s_NN; s_bin1 = s_bin; s_1 = s;
end