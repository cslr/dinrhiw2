% calculates Bernoulli-Bernoulli RBM by minimizing free energy 
% (gradient descent)

function rbm = calculate_rbm(X, NHID, CDk, EPOCHS) 

NVIS = size(X, 2);
% NHID = 10; % 100 hidden states 
% CDk  = 1;
lrate = 0.001;
% EPOCHS = 100;

a = zeros(NVIS,1);
b = zeros(NHID,1);
W = 0.1*randn(NVIS,NHID);

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


    g_pa = g_pa / N;
    g_pb = g_pb / N;
    g_pW = g_pW / N;
    
				% updates parameters by selecting 
				% a single random point
    v = X(index,:);
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

rbm.W = W;
rbm.a = a;
rbm.b = b;
rbm.CDk = CDk;
rbm.type = 'BB';



