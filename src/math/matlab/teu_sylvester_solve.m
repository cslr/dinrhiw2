
% solves sylvester equations AX - XB = C with
% bartels-steward algorithm
%


function [C] = teu_sylvester_solve(A,B,C)

[p,p] = size(A);
[r,r] = size(B);

% first iteration k = 1
k = 1;

if(B(k+1,k)) % must solve two vectors at once (2x2 block)
    H = zeros(2*p,1);
    H(1:p,1)     = C(1:p,k);
    H(p+1:2*p,1) = C(1:p,k+1);
        
    AA = zeros(2*p,2*p);
    AA(1:p,1:p)     = A - B(k,k)*eye(p,p); AA(1:p,p+1:2*p)     = -B(k+1,k)*eye(p,p);
    AA(p+1:2*p,1:p) = -B(k,k+1)*eye(p,p);  AA(p+1:2*p,p+1:2*p) = A - B(k+1,k+1)*eye(p,p);
       
    H = inv(AA)*H;
    C(1:p,k) = H(1:p,1);
    C(1:p,k+1) = H(p+1:2*p,1);
        
    k = k + 2;
else % upperdiagonal part (1x1 block)
        
    % b = c_k + SUM(g_ik * z_i);
    C(1:p,k) = C(1:p,k);
    
    % solves (A-b_kkI)b = z
    AA = A - B(k,k)*eye(p,p);
    z = inv(AA)*C(1:p,k);
    
    C(1:p,k) = z; % k:th column of X, X(:,k) = z;
    k = k + 1;
end


% solves inner parts of the solution matrix
while(k <= r-1)
    
    if(B(k+1,k)) % must solve two vectors at once (2x2 block)
        H = zeros(2*p,1);
        H(1:p,1)     = C(1:p,k)   + C(1:p,1:k-1)*B(1:k-1,k);
        H(p+1:2*p,1) = C(1:p,k+1) + C(1:p,1:k-1)*B(1:k-1,k+1);
        
        AA = zeros(2*p,2*p);
        AA(1:p,1:p)     = A - B(k,k)*eye(p,p); AA(1:p,p+1:2*p)     = -B(k+1,k)*eye(p,p);
        AA(p+1:2*p,1:p) = -B(k,k+1)*eye(p,p);  AA(p+1:2*p,p+1:2*p) = A - B(k+1,k+1)*eye(p,p);
        
        H = inv(AA)*H;
        C(1:p,k) = H(1:p,1);
        C(1:p,k+1) = H(p+1:2*p,1);
        
        k = k + 2;
    else % upperdiagonal part (1x1 block)
        
        % b = c_k + SUM(g_ik * z_i);
        C(1:p,k) = C(1:p,k) + C(1:p,1:k-1)*B(1:k-1,k);
    
        % solves (A-b_kkI)b = z
        AA = A - B(k,k)*eye(p,p);
        z = inv(AA)*C(1:p,k);
    
        C(1:p,k) = z; % k:th column of X, X(:,k) = z;
        k = k + 1;
    end
end


% last iteration (if needed)

if(k < r)
    % b = c_k + SUM(g_ik * z_i);
    C(1:p,k) = C(1:p,k) + C(1:p,1:k-1)*B(1:k-1,k);
        
    % solves (A-b_kkI)b = z
    AA = A - B(k,k)*eye(p,p);
    z = inv(AA)*C(1:p,k);
       
    C(1:p,k) = z; % k:th column of X, X(:,k) = z;
    k = k + 1;
end

