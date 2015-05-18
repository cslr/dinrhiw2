% calculates Gaussian-Bernoulli RBM by minimizing free energy 
% (gradient descent)
function rbm = calculate_gbrbm(X, NHID, CDk, EPOCHS)

NVIS = size(X, 2);
% NHID = 10; % 10 hidden states 
% CDk  = 1;
lrate = 0.001;
% EPOCHS = 100;

a = zeros(NVIS,1);
b = zeros(NHID,1);
W = 0.1*randn(NVIS,NHID);
C = eye(NVIS,NVIS); % initially we assume diagnonal covariance matrix
                    % WE DON'T CURRENTLY UPDATE C but check things work with I matrix..

Y = reconstruct_gbrbm_data(X, W, a, b, C, CDk);
err = norm(X - Y, 'fro')/prod(size(X));
printf("EPOCH %d/%d. Reconstruction error: %f\n", 0, EPOCHS, err);
fflush(stdout);

for e=1:EPOCHS

  for i=1:length(X)
				% goes through data points and 
				% calculates gradients
    g_pa = zeros(size(a));
    g_pb = zeros(size(b));
    g_pW = zeros(size(W));
    g_pC = zeros(size(C));
    N = 0;

      index = floor(rand*length(X));
      if(index < 1)
	index = 1;
      end
      v = X(index,:);
      v = reconstruct_gbrbm_data(v, W, a, b, C, CDk); % gets p(v) from the model
      cv = ((C**-0.5) * v')';
      
      h = sigmoid((cv * W)' + b);
      
      g_pa = g_pa + (C**-1)*(v' - a);
      g_pb = g_pb + h;
      g_pW = g_pW + cv' * h';
      term = + (C**-1.5)*(v'-a)*(v'-a)' - ((C**-1)*v')*(W*h)';
      g_pC = g_pC + term;
      N = N + 1;


    g_pa = g_pa / N;
    g_pb = g_pb / N;
    g_pW = g_pW / N;
    
				% updates parameters
    v = X(index,:);
    cv = ((C**-0.5) * v')';
    h = sigmoid((cv * W)' + b);

    a = a + lrate*((C**-1)*(v' - a)  - g_pa);
    b = b + lrate*(h     - g_pb);
    W = W + lrate*(cv'*h' - g_pW);
    term = + (C**-1.5)*(v'-a)*(v'-a)' - ((C**-1)*v')*(W*h)';
    
    % finds good learning rate for C matrix
    lrate_c = lrate;
    C0 = C;
    
    while(1)
      C = C0 + lrate_c*(term - g_pC);
      C = real(C);
      
      % checks if matrix has NaN or Inf values
      if(sum(sum(isnan(C))) < 1 && sum(sum(isinf(C))) < 1)
	% .. no
	
	% checks if eigenvalues of matrix are positive
	if(min(eig(C) > 0) == 1)
	  % all eigenvalues are positive
	  
	  % checks diagonal terms are positive
	  if(min(diag(C) > 0) == 1)
	    break; % we exit the lrate loop here
	  end
	end
      end
      
      C
      lrate_c = lrate_c / 2;
    end
      
    C = diag(diag(C)); % only keeps diagonal terms [C is always diagnonal..]
  end
  
				% reconstructs RBM data and 
				% calculates reconstruction error
  Y = reconstruct_gbrbm_data(X, W, a, b, C, CDk);
  err = norm(X - Y, 'fro')/prod(size(X));
  
  my = mean(Y);
  
  printf("EPOCH %d/%d. Reconstruction error: %f ", e, EPOCHS, err);
  printf("mean: %f %f ", my(1), my(2));
  printf("mean-cov: %f\n", mean(diag(C)));
  fflush(stdout);
  
end

rbm.W = W;
rbm.a = a;
rbm.b = b;
rbm.CDk = CDk;
rbm.C = C;
rbm.type = 'GB';



