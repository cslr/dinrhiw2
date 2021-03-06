% testing RBM generation

generate_test_data2

cov(X)
% X = 10*X;
% cov(X)

% rbm = calculate_gbrbm(X, 10, 1, 50);
rbm = calculate_gbrbm(X, 100, 1, 50);


Y = reconstruct_gbrbm_data(X, rbm.W, rbm.a, rbm.b, rbm.C, rbm.CDk);

figure(1);
hold off;
plot(X(:,1), X(:,2), 'go');
hold on;
plot(Y(:,1), Y(:,2), 'bo');


