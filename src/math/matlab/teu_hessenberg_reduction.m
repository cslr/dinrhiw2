% reduction to hessenberg form by
% using householder rotations
%
% Q' A Q = H

function [A,Q] = teu_hessenberg_reduction(A)

[n,m] = size(A);
Q = eye(n,n);

for k=1:n-2
  v(1:k) = 0;
  v(k+1:n) = teu_house(A(k+1:n, k));
  A(k+1:n,k:n) = teu_rowhouse(A(k+1:n,k:n), v(k+1:n));
  A(1:n, k+1:n) = teu_colhouse(A(1:n, k+1:n),transpose(v(k+1:n)));
  
  Q = teu_colhouse(Q, transpose(v));
  
end
