% 
% logarithm with the first method: n ~ 10^5 ! for 'normal' numbers!
% 

function y = m1_log(x)

sign = 1;

if(x > 1)
    x = 1/x;
    sign = -1;
end

x = 1 - x;

P = 64;


if(x == 0)
    n = 0;
else
    % we need to know order of x ~ 2^-N (typically x E [0.0, 0.5], N E [1, inf])
    % (input x from is m2_log2  is 2 (easy) or 0.5 <= x <= 1 => ok after invert.
    
    % here we can again use x = d*2^-N decomposition and use N. (from GMP)
    
    N = -log2(x)  % hard to calculate 
    n = round(P/N + 1.5);
end


y = 0;
for i=1:n
    y = y - power(x, i)/i;
end

y = sign*y;
n
