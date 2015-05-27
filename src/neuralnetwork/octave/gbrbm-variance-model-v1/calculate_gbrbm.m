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
z = zeros(NVIS, 1); % initially we assume diagonal covariance matrix I
C = diag(exp(z)); % creates temp. covariance matrix out of zetas

Y = reconstruct_gbrbm_data(X, W, a, b, C, CDk);
err = norm(X - Y, 'fro')/prod(size(X));
printf("EPOCH %d/%d. Reconstruction error: %f\n", 0, EPOCHS, err);
fflush(stdout);

for e=1:EPOCHS

  for i=1:length(X)
  
    if(sum(sum(isnan(z))) > 1)
      z
    end
    
    if(sum(sum(isnan(z))) > 1)
      a
    end
    
    if(sum(sum(isnan(z))) > 1)
      b
    end
    
    if(sum(sum(isnan(W))) > 1)
      W
    end
  
      % goes through data points and calculates gradients
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
      
      C = diag(exp(z)); % creates covariance matrix out of zetas
      Cinv05 = diag(exp(-0.5*z)); % C**-0.5
      Cinv = diag(exp(-z)); % C** -1.0
      
      v = reconstruct_gbrbm_data(v, W, a, b, C, CDk); % gets v ~ p(v) from the model
      cv = (Cinv05 * v')';
      
      h = sigmoid((cv * W)' + b);
      
      g_pa = g_pa + (Cinv)*(v' - a);
      g_pb = g_pb + h;
      g_pW = g_pW + cv' * h';
      
      term = zeros(size(g_pz));
      Wh = W*h;      
      for i=1:length(z)
        term(i) = +0.5*exp(-z(i)) * (v(i)-a(i))*(v(i)-a(i)) - 0.5*exp(-z(i)/2)*v(i)*Wh(i);
      end
      g_pz = g_pz + term;
      
      N = N + 1;


    g_pa = g_pa / N;
    g_pb = g_pb / N;
    g_pW = g_pW / N;
    
				% updates parameters
    v = X(index,:);
    cv = (Cinv05 * v')';
    h = sigmoid((cv * W)' + b);

    term = zeros(size(g_pz));
    Wh = W*h;
    for i=1:length(z)
      term(i) = +0.5*exp(-z(i)) * (v(i)-a(i))*(v(i)-a(i)) - 0.5*exp(-z(i)/2)*v(i)*Wh(i);
    end

    a = a + lrate*((Cinv)*(v' - a)  - g_pa);
    b = b + lrate*(h     - g_pb);
    W = W + lrate*(cv'*h' - g_pW);    
    z = z + lrate*(term - g_pz);
    
  end
  
				% reconstructs RBM data and 
				% calculates reconstruction error
  C = diag(exp(z)); % creates covariance matrix out of zetas
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
rbm.C = diag(exp(z));
rbm.type = 'GB';



