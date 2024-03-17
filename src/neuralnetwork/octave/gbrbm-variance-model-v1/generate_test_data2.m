% function to generate test data for RBM
% generates continous data (circle) between [-5,5] interval and adds N(0,I) noise

X = zeros(1000, 2);

for i=1:length(X)
  angle = rand*2*pi;
  X(i,1) = 10*cos(angle);
  X(i,2) = 10*sin(angle);
end

X = X + 0.1*randn(size(X));

