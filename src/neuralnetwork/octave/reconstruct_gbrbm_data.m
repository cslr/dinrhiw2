function Y = reconstruct_gbrbm_data(X, W, a, b, z, CDk)
  Y = zeros(size(X));
  
  d = exp(-z)'; % D^-1 matrix diagonal
  dd = exp(0.5*z); % D^0.5 matrix diagonal
  
  for k=1:size(X,1)
    v = X(k,:);
    
    for l=1:CDk
      h = sigmoid(( (d .* v) * W)' + b);
      h = rand(size(h)) < h; % discretizes

      m = (W * h + a)';
      v = (dd .* randn(size(m))')' + m;
    end
    
    Y(k,:) = v;
  end
  
endfunction
