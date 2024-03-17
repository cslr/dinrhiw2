%
% calculates (potentially complex)
% eigenvalues and vectors of 2x2 matrix directly
%
% (so from real schur form one can compute
%  all the eigenvalues directly)
% (trivial/own idea)

% solution from EQs:
% A(1,1) + A(2,2) = eig1 + eig2 (trace)
% A(1,1)*A(2,2) - A(2,1) * B(1,2) = eig1 * eig2 (det)
% 
% A = X*D*inv(X)

function [X,D] = teu_2dmatrix_eig(A)

% mean and "subtraction" of trace
d1 = (A(1,1) + A(2,2)) * 0.5;
d2 = (A(1,1) - A(2,2)) * 0.5;

d = sqrt( d2 * d2 + A(2,1) * A(1,2) );

e(1) = d1 - d;
e(2) = d1 + d;
D = diag(e);
n = ones(2,1);

% calculates eigenvectors
X = zeros(2,2);

if(d) % different eigenvalues (full rank - unless zero eigenvalues)
    
    for(i=1:2)
        if(A(1,2))
            X(1,i) = 1;
            X(2,i) = (- A(1,1) + e(i))/A(1,2);
        elseif(A(2,1))
            X(1,i) = (- A(2,2) + e(i))/A(2,1);
            X(2,i) = 1;
        else
            if(abs(A(1,1) - e(i)) > 0)
                X(1,i) = 0;
                X(2,i) = 1;
            else
                X(1,i) = 1;
                X(2,i) = 0;
            end
        end
        
        X(:,i) = X(:,i) / norm(X(:,i));
    end
    
else  % same eigenvalues
    
    % eigenvector 1
    if(A(1,2))
        X(1,1) = 1;
        X(2,1) = (- A(1,1) + e(1))/A(1,2);
    elseif(A(2,1))
        X(1,1) = (- A(2,2) + e(1))/A(2,1);
        X(2,1) = 1;
    else
        % both off diagonals are zero -> X = I
        X = eye(2,2);
        return;
    end
        
    X(:,1) = X(:,1) / norm(X(:,1));
    
    % solves eigenvector 2
    
    if(A(1,2))
        X(1,2) = 1;
        X(2,2) = (- A(1,1) + e(2))/A(1,2);
    elseif(A(2,1))
        X(1,2) = (- A(2,2) + e(2))/A(2,1);
        X(2,2) = 1;
    else
        X(1,2) = 0;
        X(2,2) = 1;
    end
        
    X(:,2) = X(:,2) / norm(X(:,2));    
    
end

