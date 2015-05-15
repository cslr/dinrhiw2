% testing RBM generation

generate_test_data
calculate_gbrbm

figure(1);
hold off;
plot(X(:,1), X(:,2), 'go');
hold on;
plot(Y(:,1), Y(:,2), 'bo');


