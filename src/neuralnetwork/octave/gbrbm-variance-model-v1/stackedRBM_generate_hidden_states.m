% generates hidden states of the stacked RBM

function H = stackedRBM_generate_hidden_states(X, stack)

DEEPNESS = length(stack);
[NUMDATA, INPUTDIM] = size(X);

for n=1:NUMDATA
	v = X(n, :); % visible state

	for d=1:DEEPNESS
		rbm = stack{d}.rbm;

		if(rbm.type == 'BB')
		  % Bernoulli model
                  h = rbm.W' * v' + rbm.b;
		  h = rand(size(h)) < h; % discretization
		  v = h';
                else
		  % Gaussian model
		  cv = ((rbm.C**-0.5) * v')';
                  h = rbm.W' * cv' + rbm.b;
		  h = rand(size(h)) < h; % discretization
		  v = h';
                end
	end

	H(n,:) = h;
end


