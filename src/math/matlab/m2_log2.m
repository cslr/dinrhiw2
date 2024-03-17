% 
% logarithm with the second method: n ~ 10^5 ! for 'normal' numbers!
% 

function y = m2_log2(x)

% calculating ex and d are supported by GMP (d E [0.5, 1] -> very nice for m1_log)
ex = round(log2(x));
d = x / power(2,ex)

s = 1.0/m1_log(2);  % only calculated once per base/precision

y = ex + m1_log(d)*s;


% alternative we may use natural basis and calculate:
% log(d*2^ex) = m1_log(d) + ex*m1_log(2)  -> doesn't change anything
