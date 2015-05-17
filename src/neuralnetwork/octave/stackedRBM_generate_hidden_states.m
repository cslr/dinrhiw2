% generates hidden states of the stacked RBM

function H = stackedRBM_generate_hidden_states(X, stack)

alpha = 5;

DEEPNESS = length(stack);
[NUMDATA, INPUTDIM] = size(X);

for n=1:NUMDATA
	v = X(n, :); % visible state

	for d=1:DEEPNESS
		% generates hidden state from the visible state
		rbm = stack{d}.rbm;
		h = rbm.W' * v' + rbm.b;
		h = rand(size(h)) < h; % discretization
		v = alpha*h'; % next layers visible state is this layer's hidden state + scaling
	end

	H(n,:) = h;
end


