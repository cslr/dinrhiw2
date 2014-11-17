% row house

function B = teu_rowhouse(A,v)

% s = size(A);
%
% B = eye(s(1),s(1)) - 2 * v' * v /(v * v');
% B = B*A;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w = -2 * conj(v) * A / (v * v');
B = A +  transpose(v) * w;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%w = -2 * v * A / (v * v');
%B = A +  v' * w;
