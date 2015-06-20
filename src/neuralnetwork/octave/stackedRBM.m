% calculates stacked Gaussian-Bernoulli RBMs deep network
function stack = stackedRBM(X, arch)

if(length(arch) < 2)
	stack = [];
	return;
end

for i=1:(length(arch)-1)
	printf("Training %d x %d RBM\n", arch(i), arch(i+1));
	fflush(stdout);
	
	% trains new RBM (for this layer)
	if(i == 1)
		rbm = calculate_gbrbm(X, arch(i+1), 1, 10);
	else
		rbm = calculate_rbm(X, arch(i+1), 1, 50);
	end

	stack{i}.rbm = rbm;
	
	% recalculates hidden states from the input for the next layer
	% Y = reconstruct_gbrbm_data(X, rbm.W, rbm.a, rbm.b, rbm.CDk);
	Y = X;

	clear Z;

	for k=1:size(Y, 1)
		v = Y(k,:);
		h = sigmoid((v * rbm.W)' + rbm.b);
h = rand(size(h)) < h; % discretizes next layer's data.
		Z(k,:) = h; % rescales hidden layer data for the input layer
	end
	
	X = Z;
end


