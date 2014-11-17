% calculates francis double shift reduction

function [H,Z] = francis_qr_step(H)

B = H;
[n, m] = size(H);
m = m - 1;

Z = eye(size(H));


    % implicit double shift strategy
    % with householder coefficients
    
    s = H(m,m) + H(n,n);
    t = H(m,m)*H(n,n) - H(m,n)*H(n,m);
    
    w = zeros(1, 3);
    w(1) = H(1,1)*H(1,1) + H(1,2)*H(2,1) - s*H(1,1) + t;
    w(2) = H(2,1)*( H(1,1) + H(2,2) - s );
    
    if(n > 2)
        w = zeros(1, 3);
        w(1) = H(1,1)*H(1,1) + H(1,2)*H(2,1) - s*H(1,1) + t;
        w(2) = H(2,1)*( H(1,1) + H(2,2) - s );
        w(3) = H(2,1)*H(3,2);
        
        k = 0;
        v = zeros(1, n);
        v(k+1:k+3) = teu_house(w);
    
        H(k+1:k+3,k+1:n) = teu_rowhouse(H(k+1:k+3,k+1:n),v(k+1:k+3));
        r = min(k+4,n);
        H(1:r,k+1:k+3) = teu_colhouse(H(1:r,k+1:k+3),transpose(v(k+1:k+3)));
        
        Z(k+1:k+3,k+1:n) = teu_rowhouse(Z(k+1:k+3,k+1:n),v(k+1:k+3));
        
    else
        w = zeros(1, 2);
        w(1) = H(1,1)*H(1,1) + H(1,2)*H(2,1) - s*H(1,1) + t;
        w(2) = H(2,1)*( H(1,1) + H(2,2) - s );
        
        k = 0;
        v = zeros(1, n);
        v(k+1:k+2) = teu_house(w);
    
        H(k+1:k+2,k+1:n) = teu_rowhouse(H(k+1:k+2,k+1:n),v(k+1:k+2));
        r = min(k+4,n);
        H(1:r,k+1:k+2) = teu_colhouse(H(1:r,k+1:k+2),transpose(v(k+1:k+2)));
        
        Z(k+1:k+2,k+1:n) = teu_rowhouse(Z(k+1:k+2,k+1:n),v(k+1:k+2));
    end
    
    
    % fix things back to normal
    % (to heisenberg form with the help
    % of householder rotations)
    for k=1:n-2
        % note: should only alter smaller submatrix
        % (certain rows are known to be zero already,
        %  double shift doesn't set all values to be nonzero)
        
        v(k) = 0;
        v(k+1:n) = teu_house(H(k+1:n, k));
        H(k+1:n,k:n) = teu_rowhouse(H(k+1:n,k:n), v(k+1:n));
        H(1:n, k+1:n) = teu_colhouse(H(1:n, k+1:n),transpose(v(k+1:n)));
        
        Z = teu_colhouse(Z, transpose(v));
    end




