function h = calculateHidden(v, W)
  
  h = W*v;
  h = h + randn(size(h))*0.01; % noise term as recommended in the paper?
  
  h = 2 ./ (1 + exp(-h)) - 1;
  h(length(h)) = 1;
  
  endfunction
