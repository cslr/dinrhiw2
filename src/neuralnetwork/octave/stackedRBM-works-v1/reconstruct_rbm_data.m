function Y = reconstruct_rbm_data(X, W, a, b, CDk)
  Y = zeros(size(X));
  
  for k=1:size(X,1)
    v = X(k,:);
    
    for l=1:CDk
      h = sigmoid((v * W)' + b);
      h = rand(size(h)) < h; % discretizes
      v = sigmoid(W * h + a)';
      v = rand(size(v)) < v; % DISCRETIZES VISIBLE STATES
    end
    
    Y(k,:) = v;
  end
  
endfunction
