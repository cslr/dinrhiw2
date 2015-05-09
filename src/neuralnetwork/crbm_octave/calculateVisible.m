function v = calculateVisible(h, W)
  
  v = W'*h;
  v = v + randn(size(v))*0.01; % noise term as recommended in the paper?
  
  v = 2 ./ (1 + exp(-v)) - 1;
  v(length(v)) = 1;
  
endfunction

