% calculates eigenvalues of *real* square matrix
% francis_qr_step doesn't work (yet) with complex values, when it does
% eigenvalue calculation should work with complex numbers also.
%
% calculates real schur form: A = X*T*X' , T is upper diagonal and X:s are
% rotations


function [X,T] = teu_schur(A)

B = A;
[N,N] = size(A);

% TO DO: 1st 'normalize' number values in a matrix
% to have roughly same order of magnitude

% calculates first hessenberg form of A
[A,X] = teu_hessenberg_reduction(A);

% for each iteration calculates
% absolute values of subdiagonal entries
% and processes each submatrix separately
% if number of processings == 0 for on iteration,
% stops

pr = 1; % number of rotations done in a latest iteration
sd = ones(N - 1,1);

% tolerance - when subdiagonal entry is small enough
% should be estimated from the matrix
tolerance = 0.0001 * 0.00001; % hardcoded - for now

while(pr > 0)
  pr = 0;
  
  % calculates absolute value of subdiagonal is,
  % zero or not, set values < tolerance to be zero
  for i=1:(N-1)
    if(abs(A(i+1,i)) < tolerance)
      A(i+1,i) = 0;
      sd(i) = 0;
    end
  end
  
  % finds remaining submatrices >= 3x3
  
  i = 1;
  while(i <= N - 2)
    
    % long enough block
    if(sd(i) > 0 & sd(i+1) > 0)
      sd_len = 1;      
      
      for j=i+1:N-1
        if(sd(j) > 0)
          sd_len = sd_len + 1;
        else
          break;
        end
      end
      
      % processes the block with francis qr step
      [A(i:(i + sd_len),i:(i + sd_len)), Z] = francis_qr_step(A(i:(i + sd_len),i:(i + sd_len)));
      
      % updates X
      X(i:i+sd_len,i:i+sd_len) = X(i:i+sd_len,i:i+sd_len) * Z;
      
      if(i > 1)
        X(1:(i-1),i:i+sd_len) = X(1:(i-1),i:i+sd_len) * Z;
        A(1:(i-1),i:i+sd_len) = A(1:(i-1),i:i+sd_len) * Z;
      end
      
      if(i + sd_len < N)
        X((i+sd_len+1):N,i:i+sd_len) = X((i+sd_len+1):N,i:i+sd_len) * Z;
        A(i:i+sd_len,(i+sd_len+1):N) = Z.' * A(i:i+sd_len,(i+sd_len+1):N);
      end
      
      pr = pr + 1;
      i = i + sd_len;
    end
    
    i = i + 1;
  end % while(i <= N - 2)
  
end % while(pr > 0)


% calculates eigenvalues from diagonal block matrices now
% 1x1 = directly correct eigenvalue, 
% 2x2 = calculate block's eigenvalues with 2d matrix eigenvalue solver

%
%e = zeros(N,1);
%
%j = 1; i = 1;
%while(i < N - 1)
%  if(A(i+1,i) == 0)
%    e(j) = A(i,i);
%    j = j + 1;
%    i = i + 1;
%  else
%    [e(j), e(j+1)] = teu_2dmatrix_eig(A(i:i+1,i:i+1));
%    j = j + 2;
%    i = i + 2;
%  end
%end
%
%if(i == N - 1)
%  if(A(i+1,i) == 0)
%    e(j) = A(i,i);
%    e(j+1) = A(i+1,i+1);
%  else
%    [e(j), e(j+1)] = teu_2dmatrix_eig(A(i:i+1,i:i+1));
%  end
%elseif(i == N)
%  e(j) = A(i,i);
%end 
%
% D = diag(e);

T = A;

