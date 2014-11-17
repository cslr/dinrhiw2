%
% calculates fast givens transformation
% a and b are coefficients and t = type of the matrix

function [a,b,t] = teu_fastgivens(x,d)

if x(2) ~= 0
  a = - x(1) / x(2);
  b = - a * d(2) / d(1);
  g = - a * b;
  if g <= 1
    t = 1;
    th = d(1);
    d(1) = (1 + g)*d(2);
    d(2) = (1 + g)*th;
  else
    t = 2;
    a = 1/a;
    b = 1/b;
    g = 1/g;
    
    d(1) = (1 + g)*d(1);
    d(2) = (1 + g)*d(2); 
  end
else
  t = 2;
  a = 0;
  b = 0;
end