
function y = m2_exp(x)

x = x/2;

P = 64;

if(x == 0)
    n = 0;
else
    % we need to know order of x ~ 2^-N (typically x E [0.0, 0.5], N E [1, inf])
    % (input x from is m2_log2  is 2 (easy) or 0.5 <= x <= 1 => ok after invert.
    
    % here we can again use x = d*2^-N decomposition and use N. (from GMP)
    
    N = -log2(x);  % hard to calculate (not so (upperbound: see above)) 
    n = round(P/N + 1.5);
    
    n = abs(n)  % abs not needed if x < 2
end

y = 0;
b = 1;
z = 1;
for i=1:n
    b = b/i;
    z = z*x;
    
    y = y + b*z;
end

y = 1 + 2*y + y*y;
