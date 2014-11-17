% calculates corresponding eigenvectors X(:,1) X(:,2) .. 
% for given eigenvalues e(1), e(2), ..
% each eigenvalue e(i) has n(i) eigenvectors assiciated with it
%
% note: if invpower method with gram-schimdt orthonormalization
% could be used with symmetric matrix for whole eigenvector basis
% between iterations (eigenvectors of A are orthogonal)
% (but invpower method idea is slow for *full* eigenvector problem)

function X = teu_invpowermethod(A, e, n)

[N,N] = size(A);

X = zeros(N,N);
I = sqrt(-1);


k = 1;
for i=1:length(e)
    % solves eigenvectors of i:th eigenvalue
    B = A - e(i)*eye(N,N);
    
    % implementation if B is numarically singular:
    % try to invert B, if it doesn't then set a small constant
    % to the zero position so that the 'inversion' can be done
    % find smallest
    
    invB = pinv(B);
    
    % 1st iteration: random init + normalize
    for j=0:(n(i) - 1)
        if(isreal(B))
            X(:,k+j) = invB * rand(N,1);
        else
            X(:,k+j) = invB * (rand(N,1) + rand(N,1)*I);
        end
    end
    if(n(i) > 1) X(:,k + n(i) - 1) = teu_gramschimdt(X(:,k + n(i) - 1));
    else
        for j=0:(n(i) - 1)
            X(:,k+j) = X(:,k+j) / norm(X(:,k+j));
        end
    end

    % does only one extra iteration
    % n:th iteration
    X(:,k:k+n(i)-1) = invB * X(:,k:k+n(i)-1);
    if(n(i) > 1) X(:,k + n(i) - 1) = teu_gramschimdt(X(:,k + n(i) - 1));
    else
        for j=0:(n(i) - 1)
            X(:,k+j) = X(:,k+j) / norm(X(:,k+j));
        end
    end
    
    k = k + n(i);
end


