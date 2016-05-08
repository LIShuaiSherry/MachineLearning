function svmFinance(X, y, training_percent)

total = size(X,1);
train_size = floor(total * training_percent);
%test_size = total - train_size;
trainX = X(1:train_size,:);
trainy = y(1:train_size,:);
testX = X(train_size+1: end, :);
testy = y(train_size+1: end, :);

sigma2 = 1:5:101;
g = 1.0 ./ sigma2;
%c = 1:5:101;
%d = 1:5;
train_accuracy = zeros(21,1);
test_accuracy = zeros(21,1);

for i=1:21
    for j=1
        display(sigma2(i));
        display(g(i));
        %display(c(j));
        %s = ['-t 0 -cost ', num2str(c(j))];
        s = ['-t 2 -g ', num2str(g(i)), ' -cost 78'];
        %s = ['-t 2 -g ', num2str(g(i)), ' -cost ', num2str(c(j))];
        model = svmtrain(trainy, trainX, s);
        [~, a, ~] = svmpredict(trainy, trainX, model);
        train_accuracy(i) = a(1);
        [~, a, ~] = svmpredict(testy, testX, model);
        test_accuracy(i) = a(1);
    end
end

plot(sigma2, train_accuracy, '-sk', sigma2,test_accuracy,'-^r');
title('The results of SVMs with various \sigma^2 where C = 78.');
xlabel('\sigma^2');
ylabel('Hit ratio');
legend('Training', 'Test');

end