
function y = m1_exp(x)

% x = d*2^N

% dd = x/2^N (fast to get real d fraction)
% Exp[x] = (Exp[dd])^(2^N)

ex = round(log2(abs(x)));
d = x / power(2,ex);

% so we need to compute only
% Exp[d], d E [0.5, 1]

y = power(m2_exp(d), power(2,ex));




