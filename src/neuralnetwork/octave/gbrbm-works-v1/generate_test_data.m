% function to generate test data for RBM
% generates continous data (circle) between [0,1] interval
% this can be later discretized if wanted

X = zeros(1000, 2);

for i=1:length(X)
  angle = rand*2*pi;
  X(i,1) = 5*cos(angle);
  X(i,2) = 5*sin(angle);
end

X = X + randn(size(X));

