% tests stacked RBM implementation

clear;

generate_test_data2

arch = [2 20 4];
stack = stackedRBM(X, arch);

Z = reconstruct_stackedRBM_data(X, stack);

figure(1);
hold off;
plot(X(:,1), X(:,2), 'b.'); % prints the original data
hold on;
plot(Z(:,1), Z(:,2), 'go'); % prints the reconstructed data 


