function Y = reconstruct_gbrbm_data(X, W, a, b, CDk)
  Y = zeros(size(X));
  
  for k=1:size(X,1)
    v = X(k,:);
    
    for l=1:CDk
      h = sigmoid((v * W)' + b);
      h = rand(size(h)) < h; % discretizes

      m = (W * h + a)';
      v = randn(size(m)) + m;
    end
    
    Y(k,:) = v;
  end
  
endfunction
