% calculates Gaussian-Bernoulli RBM by minimizing free energy 
% (gradient descent)
function rbm = calculate_gbrbm(X, NHID, CDk, EPOCHS)

NVIS = size(X, 2);
% NHID = 10; % 10 hidden states 
% CDk  = 1;
lrate = 0.01;
% EPOCHS = 100;

a = zeros(NVIS,1);
b = zeros(NHID,1);
W = 0.1*randn(NVIS,NHID);
z = zeros(NVIS,1); % diagonal covariance matrix initially

alpha = 1.0;

for i=1:length(z)
  z(i) = log(1); %  variance is one initially
end

Y = reconstruct_gbrbm_data(X, W, a, b, z, CDk, alpha);
err = norm(X - Y, 'fro')/prod(size(X));
printf("EPOCH %d/%d. Reconstruction error: %f\n", 0, EPOCHS, err);
fflush(stdout);


for e=1:EPOCHS

  ga = 0;
  gb = 0;
  gW = 0;
  gz = 0;
  N  = 0;


  for i=1:length(X)
				% goes through data points and 
				% calculates gradients
    g_pa = zeros(size(a));
    g_pb = zeros(size(b));
    g_pW = zeros(size(W));
    g_pz = zeros(size(z));
    N = 0;

      index = floor(rand*length(X));
      if(index < 1)
	index = 1;
      end
      v = X(index,:);
      v = reconstruct_gbrbm_data(v, W, a, b, z, CDk, alpha); % gets p(v) from the model
      
      d = exp(-z)'; % D^-2 matrix diagonal
      
      h = sigmoid(((d .* v) * W)' + b);
      
      g_pa = g_pa + alpha*(d' .* (v' - a));
      g_pb = g_pb + h;
      g_pW = g_pW + (d' .* v') * h';
      
      wh = W*h;
      for i=1:length(z)
	g_pz(i) = g_pz(i) + (0.5*alpha*(v(i)-a(i))*(v(i)-a(i)) - v(i)*wh(i));
      end
      
      N = N + 1;


    g_pa = g_pa / N;
    g_pb = g_pb / N;
    g_pW = g_pW / N;
    
				% updates parameters
    v = X(index,:);
    h = sigmoid(((d .* v) * W)' + b); 
 
    da = lrate*(alpha*(d' .* (v' - a))  - g_pa);
    db = lrate*(h     - g_pb);
    dW = lrate*((d .* v)'*h' - g_pW);
    dz = z;

    wh = W*h;
    for i=1:length(z)
      dz(i) = lrate*exp(-z(i))*((0.5*alpha*(v(i)-a(i))*(v(i)-a(i)) - v(i)*wh(i)) - g_pz(i));
    end

    a = a + da;
    b = b + db;
    W = W + dW;
    z = z + dz;
#    N = N + 1;    
  end
  
#  a = a + ga/N;
#  b = b + gb/N;
#  W = W + gW/N;
#  z = z + gz/N;
  
  
				% reconstructs RBM data and 
				% calculates reconstruction error
  Y = reconstruct_gbrbm_data(X, W, a, b, z, CDk, alpha);
  err = norm(X - Y, 'fro')/prod(size(X));
  
  my = mean(mean(Y));
  dv = sqrt( mean(exp(z)) );
  
  printf("EPOCH %d/%d. Reconstruction error: %f ", e, EPOCHS, err);
  printf("mean: %f mean-stdev: %f\n", my, dv);
  fflush(stdout);
  
end

rbm.W = W;
rbm.a = a;
rbm.b = b;
rbm.CDk = CDk;
rbm.z = z;
rbm.type = 'GB';



