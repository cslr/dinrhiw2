% testing RBM generation

generate_test_data
rbm = calculate_rbm(X, 3, 1, 50);

Y = reconstruct_rbm_data(X, rbm.W, rbm.a, rbm.b, rbm.CDk);

figure(1);
hold off;
plot(X(:,1), X(:,2), 'go');
hold on;
plot(Y(:,1), Y(:,2), 'bo');


