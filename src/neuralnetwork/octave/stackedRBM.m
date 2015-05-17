% calculates stacked Gaussian-Bernoulli RBMs deep network
function stack = stackedRBM(X, arch)

alpha = 5;

if(length(arch) < 2)
	stack = [];
	return;
end

for i=1:(length(arch)-1)
	printf("Training %d x %d RBM\n", arch(i), arch(i+1));
	fflush(stdout);
	
	% trains new RBM (for this layer)
	rbm = calculate_gbrbm(X, arch(i+1), 1, 20);
	stack{i}.rbm = rbm;
	
	% recalculates hidden states from the input for the next layer
	% Y = reconstruct_gbrbm_data(X, rbm.W, rbm.a, rbm.b, rbm.CDk);
	Y = X;

	clear Z;

	for k=1:size(Y, 1)
		v = Y(k,:);
		h = sigmoid((v * rbm.W)' + rbm.b);
		h = rand(size(h)) < h; % discretizes next layer's data.
		Z(k,:) = alpha*h; % rescales hidden layer data for the input layer
	end
	
	X = Z;
end


