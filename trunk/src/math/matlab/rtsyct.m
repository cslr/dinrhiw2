%
% recursive sylvester equations solver as described in
% "Recursive Block Algorithms for Solving Triangular Systems - 
%  Part I: One-Sided and Coupled Sylvester-Type Matrix Equations (2002)"
%
% According to experimental results this should reduce cache misses considerably 
% -> 5-10-fold speedups with computer architechtures with deep memory hierarchy
% (modern computers with multiple levels of cache+swapping)
%
% this is for upper quasitriangular A and B
% 
% A = [MxM matrix]
% B = [NxN matrix]
% X = [MxN matrix]
% C = [MxN matrix]

function X = rtsyct(A,B,C)

blks = 4;
[M,M] = size(A);
[N,N] = size(B);

if 1 <= M & N <= blks
    X = teu_sylvester_solve(A,B,C);
else
    
    if(1 <= N & N <= M/2)
        [j,i] = rtsyct_split(A,B,1); % splits A by rows and columns
        
        X(j+1:M,:) = rtsyct(A(j+1:N,i+1:N), B, C(j+1:N,:));
        C(1:j,:) =  gemm(- A(1:j,i+1:N), X(j+1:M,:), C(1:j,:) );
        X(1:j,:) = rtsyct(A(1:j,1:i),B,C(1:j,:));
        
    elseif(1 <= M & M <= N/2)
        [j,i] = rtsyct_split(A,B,2); % splits B by rows
        
        X(:, 1:i) = rtsyct(A,B(1:j,1:i), C(:,1:i));
        C(:, i+1:N) =  gemm( X(:,1:i), B(1:j,i+1:N), C(:,i+1:N) );
        X(:, i+1:N) = rtsyct(A, B(j+1:N,i+1:N), C(:,i+1:N));
    else
        [j,i] = rtsyct_split(A,B,3); % splits A and B by rows and columns
        
        if(i == M) % cannot split (extremely rare but possible(?) case)
            X = teu_sylvester_solve(A,B,C);
        else
            X(j+1:M,1:i) = rtsyct(A(j+1:M,i+1:M),B(1:j,1:i),C(j+1:M,1:i));
            C(j+1:M,i+1:N) = gemm(X(j+1:M,1:i), B(1:j,i+1:N), C(j+1:M,i+1:N));
            C(1:j,1:i) = gemm(-A(1:j,i+1:M),X(j+1:M,1:i),C(1:j,1:i));
            X(j+1:M,i+1:N) = rtsyct(A(j+1:M,i+1:M),B(j+1:N,i+1:N),C(j+1:M,i+1:N));
            X(1:j,1:i) = rtsyct(A(1:j,1:i), B(1:j,1:i), C(1:j,1:i));
            C(1:j, i+1:N) = gemm(-A(1:j,i+1:N),X(j+1:M,i+1:N),C(1:j,i+1:N));
            C(1:j, i+1:N) = gemm(X(1:j,1:i),B(1:j,i+1:N),C(1:j,i+1:N));
            X(1:j, i+1:N) = rtsyct(A(1:j,1:i),B(j+1:N,i+1:N),C(1:j,i+1:N));
        end
    end
    
end

