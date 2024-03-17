function Y = reconstruct_gbrbm_data(X, W, a, b, C, CDk)
  Y = zeros(size(X));
  
  for k=1:size(X,1)
    v = X(k,:);
    
    for l=1:CDk
      Cinv05 = diag(1 ./ sqrt(diag(C))); % C**-0.5
      cv = (Cinv05 * v')';
      h = sigmoid((cv * W)' + b);
      h = rand(size(h)) < h; % discretizes

      m = ((Cinv05) * W * h + a)';
      % assumes matrix is diagonal
      A = diag(sqrt(abs(diag(C))));
%      [V, L] = eig(C);
%      A = V*(L**0.5);
      v = (A*randn(size(m))')' + m;
    end
    
    Y(k,:) = v;
  end
  
endfunction
