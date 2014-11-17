% performs fast row givens rotation
% a and b are values of givens matrix
% t is type of matrix

function A = teu_rowfastgivens(A,a,b,t)

[p,q] = size(A);

if(t == 1)
  for j=1:q
    th1 = A(1,j);
    th2 = A(2,j);
    
    A(1,j) = b*th1 + th2;
    A(2,j) = th1 + a*th2;
  end
else
  for j=1:q
    th1 = A(1,j);
    th2 = A(2,j);
    
    A(1,j) = th1 + b*th2;
    A(2,j) = a*th1 + th2;
  end
end
    