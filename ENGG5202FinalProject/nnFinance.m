function [train_accuracy, test_accuracy] = nnFinance(X, y, training_percent)

batch_size = 100;

total = size(X,1);
train_size = floor(total * training_percent /batch_size) * batch_size;
trainX = X(1:train_size,:);
trainy = y(1:train_size,:);
testX = X(train_size+1: end, :);
testy = y(train_size+1: end, :);

hidden_nodes = 6:12;
train_accuracy = zeros(7, 3);
test_accuracy = zeros(7, 3);

for i=1:7

nn = nnsetup([12 hidden_nodes(i) 2]); % to build up one three layers network
nn.activation_function = 'sigm';
nn.output= 'softmax';

nn.learningRate = 0.1;
opts.numepochs = 10000; %Number of full sweeps through data
opts.batchsize = batch_size; %Take a mean gradient step over this many %samples

nn = nntrain(nn, trainX, trainy, opts);
trainLabels = nnpredict(nn, trainX);
trainTruey = vectorToLabel(trainy);
train_accuracy(i,:) = [hidden_nodes(i), sum(trainLabels == trainTruey), sum(trainLabels == trainTruey) / size(trainX,1) * 100.0];

labels = nnpredict(nn, testX);
truey = vectorToLabel(testy);
test_accuracy(i,:) = [hidden_nodes(i), sum(labels == truey), sum(labels == truey) / size(testX,1) * 100.0];

end

display(train_size);
display(train_accuracy);
display(test_accuracy);

end

function result = vectorToLabel(y)

n=size(y,1);
result = zeros(n,1);
for i=1:n
    if y(i,:) == [1,0]
        result(i) = 1;
    else
        result(i) = 2;
    end
end


end