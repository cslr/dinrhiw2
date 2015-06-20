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

for i=1:length(z)
  z(i) = log(1); %  variance is one initially
end

Y = reconstruct_gbrbm_data(X, W, a, b, z, CDk);
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
    g_pz = zeros(size(z));
    N = 0;

      index = floor(rand*length(X));
      if(index < 1)
	index = 1;
      end
      v = X(index,:);
      v = reconstruct_gbrbm_data(v, W, a, b, z, CDk); % gets p(v) from the model
      
      d = exp(-z)'; % D^-2 matrix diagonal
      
      h = sigmoid(((d .* v) * W)' + b);
      
      g_pa = g_pa + (d' .* (v' - a));
      g_pb = g_pb + h;
      g_pW = g_pW + (d' .* v') * h';
      
      wh = W*h;
      for i=1:length(z)
	g_pz(i) = g_pz(i) + (0.5*(v(i)-a(i))*(v(i)-a(i)) - v(i)*wh(i));
      end
      
      N = N + 1;


    g_pa = g_pa / N;
    g_pb = g_pb / N;
    g_pW = g_pW / N;
    
				% updates parameters
    v = X(index,:);
    h = sigmoid(((d .* v) * W)' + b); 
 
    a1 = a + 1.0*lrate*((d' .* (v' - a))  - g_pa);
    b1 = b + 1.0*lrate*(h     - g_pb);
    W1 = W + 1.0*lrate*((d .* v)'*h' - g_pW);

    wh = W1*h;
    for i=1:length(z)
      z1(i) = z(i) + lrate*exp(-z(i))*((0.5*(v(i)-a(i))*(v(i)-a(i)) - v(i)*wh(i)) - g_pz(i));
    end
    
    a2 = a + 1.1*lrate*((d' .* (v' - a))  - g_pa);
    b2 = b + 1.1*lrate*(h     - g_pb);
    W2 = W + 1.1*lrate*((d .* v)'*h' - g_pW);

    wh = W2*h;
    for i=1:length(z)
      z2(i) = z(i) + 1.1*lrate*exp(-z(i))*((0.5*(v(i)-a(i))*(v(i)-a(i)) - v(i)*wh(i)) - g_pz(i));
    end
    
    a3 = a + 0.9*lrate*((d' .* (v' - a))  - g_pa);
    b3 = b + 0.9*lrate*(h     - g_pb);
    W3 = W + 0.9*lrate*((d .* v)'*h' - g_pW);

    wh = W3*h;
    for i=1:length(z)
      z3(i) = z(i) + 0.9*lrate*exp(-z(i))*((0.5*(v(i)-a(i))*(v(i)-a(i)) - v(i)*wh(i)) - g_pz(i));
    end
    
%    z1 = z1';
%    z2 = z2';
%    z3 = z3';

  if(size(z) != size(z1))
    z1 = z1';
  end

  if(size(z) != size(z2))
    z2 = z2';
  end

  if(size(z) != size(z3))
    z3 = z3';
  end
  
    L = 100;

    err1 = norm(X(1:L,:) - reconstruct_gbrbm_data(X(1:L,:), W1, a1, b1, z1, CDk), 'fro')/prod(size(X(1:L,:)));
    err2 = norm(X(1:L,:) - reconstruct_gbrbm_data(X(1:L,:), W2, a2, b2, z2, CDk), 'fro')/prod(size(X(1:L,:)));
    err3 = norm(X(1:L,:) - reconstruct_gbrbm_data(X(1:L,:), W3, a3, b3, z3, CDk), 'fro')/prod(size(X(1:L,:)));
    
    if(err1 <= err2 && err1 <= err3)
      a = a1;
      b = b1;
      z = z1;
      W = W1;
      lrate = 1.0*lrate;
    elseif(err2 <= err1 && err2 <= err3)
      a = a2;
      b = b2;
      z = z2;
      W = W2;
      lrate = 1.1*lrate;
    else
      a = a3;
      b = b3;
      z = z3;
      W = W3;
      lrate = 0.9*lrate;
    end
    
  end
  
				% reconstructs RBM data and 
				% calculates reconstruction error
  Y = reconstruct_gbrbm_data(X, W, a, b, z, CDk);
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



