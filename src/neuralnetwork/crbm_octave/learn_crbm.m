%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% learns Continuous RBM parameters (currently just W)

[Nsamples, Nvis] = size(X);
Nhid = 10;
NUM_EPOCHS = 25;
CD = 10; % CD-k rule
lrate = 0.01;

v = zeros(Nvis + 1, 1); % visual layer input nodes
v(Nvis+1) = 1;

h = zeros(Nhid + 1, 1); % hidden layer nodes
h(Nhid+1) = 1;

W = randn(Nhid+1, Nvis+1)*0.01; % initial weight matrix

for e=1:NUM_EPOCHS
  Wold = W;
  
  for n=1:Nsamples
    v(1:Nvis) = X(n,:); v(Nvis+1) = 1;
    h = calculateHidden(v, W); % calculates hidden layer results
    
    P = h * v'; % model estimate gradient (positive phase)
    
    for c=1:CD
      v = calculateVisible(h, W); % calculates visible layer results
      h = calculateHidden(v, W); % calculates hidden layer results
    end
    
    N = h * v'; % stimulation estimate gradient (negative phase)
    
    W = W + lrate*(P - N); % RBM contrastive divergence learning rule
  end
  
  printf("%d/%d epoch delta W = %f\n", e, NUM_EPOCHS, max(max(Wold - W)));
  fflush(stdout);
  
end






