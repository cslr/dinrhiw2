% calculates givens rotation

function [c,s] = teu_givens(a, b)

if b == 0
  c = 1; s = 0;
else
  if(abs(b) > abs(a))
    th = -a / b;
    s = 1/sqrt(1 + th*th);
    c = s*th;
  else
    th = -b / a;
    c = 1/sqrt(1 + th*th);
    s = c*th;
  end
end
