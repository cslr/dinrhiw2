% uses householder rotation to factorize
% A =  Q * C, C is lower triangular and Q is product
% of householder rotations
% the remaining part of non-triangular area
% C = A*Q' = A * (Q(n) * Q(n-1) * .. Q(1))'

function [C,Q] = teu_householder_triangelize(A)

C = A;
s = size(C);
A = zeros(s(1),s(2)); % zeroes A
n = s(1);

for j=[1:n]
  v = teu_house(C([j:n], j));
  A(j,[j:n]) = v.'; % saves rotation for Q construction 
  C([j:n],[j:n]) = teu_rowhouse(C([j:n],[j:n]), v.');
end

% calculates Q (in 'inverse' order, so full householder
% rotation for matrix must be done only once (last step)

Q = eye(n,n);

for j=[n:-1:1]
  Q([j:n],[j:n]) = teu_rowhouse(Q([j:n],[j:n]), A(j,[j:n]));
end

