% testing RBM generation

generate_test_data2
rbm = calculate_gbrbm(X, 2, 1, 20);

Y = reconstruct_gbrbm_data(X, rbm.W, rbm.a, rbm.b, rbm.CDk);

figure(1);
hold off;
plot(X(:,1), X(:,2), 'go');
hold on;
plot(Y(:,1), Y(:,2), 'bo');


