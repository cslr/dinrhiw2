%
% calculates correct location for split of A and B
%

function [j,i] = rtsyct_split(A,B,type)

[M,M] = size(A);
[N,N] = size(B);

switch type
case 1
    % splits A
    j = floor(M/2); i = j;
    
    if(j<M)
        if(A(j+1,i)) % in the middle of 2x2 block
            i = i + 1;
            j = j + 1;
        end
    end
    
case 2
    j = floor(N/2); i = j;
    
    if(j<N)
        if(B(j+1,i))
            i = i + 1;
            j = j + 1;
        end
    end
    
case 3
    j = floor(M/2); i = j;
    
    R = min(N,M);
    
    % finds split points with works with both matrices
    % this can fail -> split point = min(N,M)
    if(j<R)
        while( (B(j+1,i) | A(j+1,i)) & (j+1 < R & i < R) )
            i = i + 1;
            j = j + 1;
        end
    end
    
    % splitting failed, tries to split in the upper part of the matrix
    if(j == R | i == R)
        j = floor(M/2); i = j;
        
        while( ( B(j+1,i) | A(j+1,i) ) & (i > 1 & j > 1) )
            i = i - 1;
            j = j - 1;
        end
        
        if(i == 1 | j == 1) % splitting isn't possible -> must proces the whole matrix
            j = M;
            i = M;
        end
    end
    
end % switch


