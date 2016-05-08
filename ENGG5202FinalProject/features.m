function [ X, y ] = features( H, L, C, n )
% This function extracts features from Hang Seng index daily data.

total = size(C, 1);

M = zeros(total,1);
for t=1:total
    M(t) = mean([H(t), L(t), C(t)]);
end

y = zeros(total,1);
for t=1:total -1
    if C(t) > C(t+1)
        y(t) = 1;
    else
        y(t) = -1;
    end
end

total1 = min(total-n, total -9);

%total1 = total - n + 1;
LL = zeros(total1, 1);
HH = zeros(total1, 1);
MA5 = zeros(total1, 1);
MA10 = zeros(total - 9, 1);
SM = zeros(total1, 1);

perK = zeros(total1, 1);
for t=1:total1
    j = t + n - 1;
    LL(t) = min(L(t:j,1));
    HH(t) = max(H(t:j,1));
    SM(t) = mean(M(t:j, 1));
    MA5(t) = mean(C(t:t+4,1));
    MA10(t) = mean(C(t:t+9,1));
    
    perK(t) = (C(t) - LL(t)) * 100.0 / (HH(t) - LL(t));
end

total2 = total1 -n +1;
perD = zeros(total2,1);
D = zeros(total2, 1);
for t=1:total2
    j = t+n-1;
    perD(t) = mean(perK(t: j));
    D(t) = mean(abs(M(t:j,1) - SM(t:j,1)));
end

total3 = total2 - n;
slowPerD = zeros(total3, 1);
momentum = zeros(total3, 1);
ROC = zeros(total3, 1);
williamR = zeros(total3, 1);
AD_Oscillator = zeros(total3,1);
Disparity5 = zeros(total3, 1);
Disparity10 = zeros(total3, 1);
OSCP = zeros(total3, 1);
CCI = zeros(total3, 1);
RSI = zeros(total3, 1);
for t=1:total3
    t0 = total3 + 2 - t;
    t1 = t0 + n - 1;
    slowPerD(t) = mean(perD(t0:t1));
    momentum(t) = C(t0) - C(t0+4);
    ROC(t) = C(t0) * 100.0 / C(t0+n);
    williamR(t) = (H(t1) - C(t0)) * 100.0 / (H(t1) - L(t1));
    AD_Oscillator(t) = (H(t0) - C(t0+1)) / (H(t0) - L(t0));
    Disparity5(t) = C(t0) * 100.0 / MA5(t0);
    Disparity10(t) = C(t0) * 100.0 / MA10(t0);
    OSCP(t) = (MA5(t0) - MA10(t0))/MA5(t0);
    CCI(t) = (M(t0) - SM(t0)) / (0.015 * D(t0));
    RSI(t) = 100 - 100 / (1 + sum( y(t0:t1, 1) == 1) * sum(y(t0:t1,1)==-1) / (n*n)); 
end

perK = scaleVector(perK(total3+1:-1:2,1));
perD = scaleVector(perD(total3+1:-1:2,1));
slowPerD = scaleVector(slowPerD);
momentum = scaleVector(momentum);
ROC = scaleVector(ROC);
williamR = scaleVector(williamR);
AD_Oscillator = scaleVector(AD_Oscillator);
Disparity5 = scaleVector(Disparity5);
Disparity10 = scaleVector(Disparity10);
OSCP = scaleVector(OSCP);
CCI = scaleVector(CCI);
RSI = scaleVector(RSI);

y = y(total3+1:-1:2,1);
X = [perK, perD, slowPerD, momentum, ROC, williamR, AD_Oscillator, Disparity5, Disparity10, OSCP, CCI, RSI];

end

