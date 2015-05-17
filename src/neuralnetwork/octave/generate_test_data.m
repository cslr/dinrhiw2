% function to generate test data for RBM
% generates continous data (sphere) between [-1,1] interval
% this can be then discretized

X = zeros(1000, 3);

for i=1:length(X)
  angle = rand*2*pi;
  angle2 = rand*2*pi;
  X(i,1) = sin(angle2)*cos(angle);
  X(i,2) = sin(angle2)*sin(angle);
  X(i,3) = sin(angle2);
end

% now we discretize data to get interesting correlations
X = double((X < -0.5) | (X > +0.5));

% X = X + randn(size(X));

