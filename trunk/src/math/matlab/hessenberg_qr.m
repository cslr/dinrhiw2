%
% calculates hessenberg QR decomposition
% and next step H+ = RQ, when H = QR
% (needed in iterative QR Schur form calculation)
%
% uses givens rotations as suggested by
% Matrix Computations (Golub & Van Loan)

function H = hessenberg_qr(H)

[n,n] = size(H);

% note should use fast givens instead

for k=1:n-1
    [c(k),s(k)] = teu_givens(H(k,k),H(k + 1, k));
    H(k:k+1,k:n) = teu_rowgivens(H(k:k+1,k:n),c(k),s(k));
end

for k=1:n-1
    H(1:k+1,k:k+1) = teu_colgivens(H(1:k+1,k:k+1),c(k),s(k));
end