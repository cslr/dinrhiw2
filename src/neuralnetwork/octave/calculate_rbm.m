
% calculates RBM by minimizing free energy (gradient descent)

NVIS = 2;
NHID = 10; % 10 hidden states 
CDk  = 10;
lrate = -0.05;
EPOCHS = 100;

a = 0.01*randn(NVIS,1);
b = 0.01*randn(NHID,1);
W = 0.01*randn(NVIS,NHID);

Y = reconstruct_rbm_data(X, W, a, b, CDk);
err = norm(X - Y, 'fro')/prod(size(X));
printf("EPOCH %d/%d. Reconstruction error: %f\n", 0, EPOCHS, err);


for e=1:EPOCHS
  
  % goes through data points and calculates gradients
  g_pa = zeros(size(a));
  g_pb = zeros(size(b));
  g_pW = zeros(size(W));
  N = 0;
  for k=1:length(X)
    v = X(k,:);
    v = reconstruct_rbm_data(v, W, a, b, CDk); % gets p(v) from the model
    h = sigmoid((v * W)' + b);
    
    g_pa = g_pa + v';
    g_pb = g_pb + h;
    g_pW = g_pW + v' * h';
    N = N + 1;
  end

  g_pa = g_pa / N;
  g_pb = g_pb / N;
  g_pW = g_pW / N;
  
  % updates parameters by going through the data again
  for k=1:length(X)
    v = X(k,:);
    h = sigmoid((v * W)' + b);

    a = a + lrate*(v'    - g_pa);
    b = b + lrate*(h     - g_pb);
    W = W + lrate*(v'*h' - g_pW);
  end
  
  % reconstructs RBM data and calculates reconstruction error
  Y = reconstruct_rbm_data(X, W, a, b, CDk);
  err = norm(X - Y, 'fro')/prod(size(X));
  
  my = mean(Y);
  
  printf("EPOCH %d/%d. Reconstruction error: %f\n", e, EPOCHS, err);
  printf("Mean: %f %f\n", my(1), my(2));
  fflush(stdout);
  
end

