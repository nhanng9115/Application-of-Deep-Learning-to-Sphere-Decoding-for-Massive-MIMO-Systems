% clear all;
clc;

Nt = 8; % transmitter antenna
Nr = 8; % receiver antenna

SNR_dB_vec = 0:4:12;

% modulation
modScheme = 'QPSK';
[symset, mOrder, Emax] = loadAlphabet(modScheme);
modAlphabet_sort1 = [symset symset];

nSymbols = length(symset);
nBit = log2(mOrder);
nIter = 2000;

nElement = length(SNR_dB_vec);
simBer = zeros(nElement,1);
K = 4;
for ss = 1:length(SNR_dB_vec)
    ss
    snr_dB = SNR_dB_vec(ss);
    snr = 10^(snr_dB/10);
    nError = 0;
    
    for nn = 1:nIter
        %% Transmitter
        x = randi(mOrder,Nt,1)-1;
        H = 1/sqrt(2)*(randn(Nr,Nt) + 1i*randn(Nr,Nt));
        n = 1/sqrt(2)*(randn(Nr,1) + 1i*randn(Nr,1));
        
        %% Convert xMod to binary string
        xMod = qammod(x,mOrder,'gray'); % BPSK
        xBin = de2bi(x, nBit);
        
        %% Receiver
        Es = mean(abs(symset).^2); % % average symbol energy
        sigma2 = Nt*Es / snr; % noise variance
        y = H*xMod + sqrt(sigma2)*n;
        
        %% QR
        [Q,R] = qr(H);
        y_tilde = Q'*y;
        
        d_correct = abs(Q'*y - R*xMod).^2;
        d1_correct = sum(d_correct);
        d2_correct = d_correct(2);
        
        
        for layer = Nt:-1:1
            d_list = [];
            x_list = [];
            
            if layer == Nt
                for i = 1:nSymbols
                    x_layer = symset(i);
                    e_square = abs(y_tilde(layer) - R(layer,2)*symset(i))^2;
                    d = e_square;
                    
                    x_list = cat(2,x_list,x_layer);
                    d_list = cat(2,d_list,d);
                    1;
                end
            else
                for k = 1:K
                    x_upper_layer = x_layer_sorted(:,k);
                    
                    for i = 1:nSymbols
                        x_layer = symset(i);
                        x_tmp = [x_layer; x_upper_layer];
                        
                        e_square = abs(y_tilde(layer) - R(layer,layer:end)*x_tmp)^2;
                        d = d_layer_sorted(k) + e_square;
                        
                        x_list = cat(2,x_list,x_tmp);
                        d_list = cat(2,d_list,d);
                    end
                end
            end
            [d_layer_sorted, idx] = sort(d_list);
            x_layer_sorted = x_list(:,idx);
        end
        xHatMod = x_layer_sorted(:,1);
        %% demod signal and covert back to binary
        xHat = qamdemod(xHatMod,mOrder,'gray') ;
        xHatBin = de2bi(xHat, nBit);
        nError = nError + biterr(xBin,xHatBin);
    end
    simBer(ss)   = nError/(Nt*nIter*nBit);
    
end
semilogy(SNR_dB_vec,simBer,'r-o','LineWidth',1.5);hold on
axis([SNR_dB_vec(1) SNR_dB_vec(end) 10^-4 1])
xlabel('SNR') ;
ylabel('BER') ;
grid on;