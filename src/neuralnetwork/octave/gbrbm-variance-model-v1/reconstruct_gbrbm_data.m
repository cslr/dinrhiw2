function Y = reconstruct_gbrbm_data(X, W, a, b, C, CDk)
  Y = zeros(size(X));
  
  for k=1:size(X,1)
    v = X(k,:);
    
    for l=1:CDk
      cv = ((C**-0.5) * v')';
      h = sigmoid((cv * W)' + b);
      h = rand(size(h)) < h; % discretizes

      m = ((C**-0.5) * W * h + a)';
      [V, L] = eig(C);
      A = V*(L**0.5);
      v = (A*randn(size(m))')' + m;
    end
    
    Y(k,:) = v;
  end
  
endfunction
