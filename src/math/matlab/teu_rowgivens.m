% givens row rotation

function A = teu_rowgivens(A,c,s)

[p, q] = size(A);

for j=[1:q]
  t1 = A(1,j); t2 = A(2,j);
  A(1,j) = c*t1 - s*t2;
  A(2,j) = s*t1 + c*t2; 
end
  