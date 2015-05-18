% function to generate test data for RBM
% generates continous data (circle) between [-5,5] interval and adds N(0,I) noise

X = zeros(1000, 2);

for i=1:length(X)
  angle = rand*2*pi;
  X(i,1) = 5*cos(angle);
  X(i,2) = 5*sin(angle);
end

X = X + randn(size(X)); % adds N(0,I) noise

