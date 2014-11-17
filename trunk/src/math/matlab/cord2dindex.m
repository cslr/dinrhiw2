%
% calculates index for vector index for 2d coordinates in DxD image
%

function i = cord2dindex(a,b, D)

% a is row number
% b is col. number

i = floor(b-1) + floor(a-1)*D;

i = i + 1; % range: i >= 1