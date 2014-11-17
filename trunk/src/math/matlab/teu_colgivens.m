% givens column rotation

function A = teu_colgivens(A,c,s)

[q, p] = size(A);

for i=[1:q]
  t1 = A(i,1); t2 = A(i,2);
  A(i,1) = c*t1 - s*t2;
  A(i,2) = s*t1 + c*t2;
end


