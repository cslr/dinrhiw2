
% calculates Gaussian-Bernoulli RBM by minimizing free energy 
% (gradient descent)

NVIS = 2;
NHID = 100; % 100 hidden states 
CDk  = 2;
lrate = 0.01;
EPOCHS = 100;

a = 0.01*randn(NVIS,1);
b = 0.01*randn(NHID,1);
W = 0.01*randn(NVIS,NHID);

Y = reconstruct_rbm_data(X, W, a, b, CDk);
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
    N = 0;
    for k=1:20
      index = floor(rand*length(X));
      if(index < 1)
	index = 1;
      end
      v = X(index,:);
      v = reconstruct_rbm_data(v, W, a, b, CDk); % gets p(v) from the model
      h = sigmoid((v * W)' + b);
      
      g_pa = g_pa + (v');
      g_pb = g_pb + h;
      g_pW = g_pW + v' * h';
      N = N + 1;
    end

    g_pa = g_pa / N;
    g_pb = g_pb / N;
    g_pW = g_pW / N;
    
				% updates parameters by selecting 
				% a single random point
    k = floor(rand(length(X)));
    if(k < 1) k = 1; end
    v = X(k,:);
    h = sigmoid((v * W)' + b);

    a = a + lrate*((v')  - g_pa);
    b = b + lrate*(h     - g_pb);
    W = W + lrate*(v'*h' - g_pW);
  end
  
				% reconstructs RBM data and 
				% calculates reconstruction error
  Y = reconstruct_rbm_data(X, W, a, b, CDk);
  err = norm(X - Y, 'fro')/prod(size(X));
  
  my = mean(Y);
  
  printf("EPOCH %d/%d. Reconstruction error: %f ", e, EPOCHS, err);
  printf("mean: %f %f\n", my(1), my(2));
  fflush(stdout);
  
end

