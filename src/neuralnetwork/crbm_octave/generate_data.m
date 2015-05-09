%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% creates test data for Continuous RBM code

DATAPOINTS = 1000;
X = rand(DATAPOINTS, 2);

for i=1:length(X)
  angle = rand*2*pi;
  X(i,1) = cos(angle);
  X(i,2) = sin(angle);
end

% adds gaussian noise 0.01
X = X + randn(size(X))*0.01;

figure(1);
plot(X(:,1),X(:,2),'og');








