% col house

function B = teu_colhouse(A,v)

%s = size(A);
%
%B = eye(s(2),s(2)) - 2 * v * v' /(v' * v);
%B = A*B;

%
%b = -2 / (v' * v);
%w = b * A * v;
%B = A +  w * v';
%

w = -2 * A * v / (v' * v);
B = A +  w * v';

