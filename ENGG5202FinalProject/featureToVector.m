function result = featureToVector(y)
%UNTITLED11 Summary of this function goes here
%   Detailed explanation goes here

n = size(y,1);
result = zeros(n,2);
for i=1:n
    if y(i)==1
        result(i,:) = [1,0];
    else
        result(i,:) = [0,1];
    end
end

end
