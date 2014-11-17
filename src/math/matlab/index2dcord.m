%
% calculates 2d coordinates in DxD image from vector index
%

function [a,b] = index2dcord(i,D)

i = floor(i - 1);

a = floor(i / D);
b = mod(i, D);

a = a + 1;
b = b + 1;

%a = floor(i / D); %rows
%b = i - a*D; % columns
%
%a = a + 1;

