function [ high, low, close] = preprocessData( high, low, close)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

index = high == low;
high(index,:)=[];
low(index,:)=[];
close(index,:)=[];

end

