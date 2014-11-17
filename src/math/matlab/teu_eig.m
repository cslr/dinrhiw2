% calculates eigenvalues of *real* square matrix

function [D,X] = teu_eig(A)

[D,X] = teu_schur(A); % calculates real schur form: X' * A * X = D