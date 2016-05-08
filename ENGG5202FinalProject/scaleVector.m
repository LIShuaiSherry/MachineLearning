function y = scaleVector( x )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

y = 2*(x - min(x))/(max(x) - min(x)) - 1;

end

