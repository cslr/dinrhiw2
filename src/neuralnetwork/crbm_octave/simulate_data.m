
Y = 2*rand(200, 2) -1; % random data between [-1,1]

for n=1:length(Y)
  v(1:Nvis) = Y(n,:); v(Nvis+1) = 1;

  for c=1:CD
    h = calculateHidden(v, W);
    v = calculateVisible(h, W);
  end
  
  Y(n,:) = v(1:Nvis);
end

figure(2);
hold off;

for n=1:length(Y)
  plot(X(n,1), X(n,2), 'og');
  hold on;
end

for n=1:length(Y)
  plot(Y(n,1), Y(n,2), 'o');
  hold on;
end





