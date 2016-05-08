function results = knnFinance(X, y, training_percent)
%UNTITLED15 Summary of this function goes here
%   Detailed explanation goes here

total = size(X,1);
train_size = floor(total * training_percent);
%test_size = total - train_size;
trainX = X(1:train_size,:);
trainy = y(1:train_size,:);
testX = X(train_size+1: end, :);
testy = y(train_size+1: end, :);


results = zeros(50,3);
for k=1:50
knnModel = fitcknn(trainX,trainy,'NumNeighbors',k);
label = predict(knnModel,testX);
results(k,:) = [k, sum(label == testy), sum(label == testy) / size(testX,1) * 100.0];

end
display(results);

k=1:50;
result = results(:,3);
plot(k, result, '-^r');
title('The results of NN with different number of hidden nodes.');
xlabel('number of nearest neighbours');
ylabel('Hit ratio (%)');
%legend('Training', 'Test');

end

