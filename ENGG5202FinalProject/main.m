function main()
%project for engg5202
%prediction of finance time series data using svm, nn, knn
%   libsvm 3.21 
%   https://www.csie.ntu.edu.tw/~cjlin/libsvm/
%   deep learning toolbox 
%   https://github.com/rasmusbergpalm/DeepLearnToolbox

clear; clc;

[high, low, close] = preprocessData( xlsread('hsi','C2:C2596'), xlsread('hsi','D2:D2596'), xlsread('hsi','E2:E2596'));
n=6;
[X, y] = features(high, low, close, n);
training_percent = 0.8;

% make(); % libsvm 3.21
% svmFinance(X, y, training_percent);
% 
% nnFinance(X, featureToVector(y), training_percent);

knnFinance(X, y, training_percent);

end




