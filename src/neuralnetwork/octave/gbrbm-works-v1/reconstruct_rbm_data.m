function Y = reconstruct_rbm_data(X, W, a, b, CDk)
  Y = zeros(size(X));
  
  for k=1:size(X,1)
    v = X(k,:);
    
    for l=1:CDk
      h = sigmoid((v * W)' + b);
      h = rand(size(h)) < h; % discretizes only hidden states!
      v = sigmoid(W * h + a)';
      if(l != CDk) v = rand(size(v)) < v; end; % DISCRETIZES VISIBLE STATES???
    end
    
    Y(k,:) = v;
  end
  
endfunction
