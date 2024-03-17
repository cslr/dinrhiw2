% reconstructs data X using stacked RBM

function Y = reconstruct_stackedRBM_data(X, stack)

% first we generate hidden states from input data
H = stackedRBM_generate_hidden_states(X, stack);

% then we generate visible states from hidden states
Y = stackedRBM_generate_data(H, stack);

% done
