% generates data from the stacked RBMs using hidden states DATA (for the top layer)

function X = stackedRBM_generate_data(DATA, stack)

% we generate random points for the top layer of the RBM
% and then generate the visible layer from that state

alpha = 5;

DEEPNESS = length(stack);

NUMDATA = size(DATA, 1);

% random [0,5] hidden states of the top layer
% DATA = rand(NUMDATA, length(stack{DEEPNESS}.rbm.b));
DATA = rand(size(DATA)) < DATA; % DISCRETIZES THEM
DATA = alpha*DATA;

for n=1:NUMDATA
	h = DATA(n, :)' / alpha;
	
	% now we backward iterate back to 
	% the visible layer from the top layer
	for d=DEEPNESS:-1:1
		rbm = stack{d}.rbm;
		v = (rbm.W*h + rbm.a);
		% do not add N(0,I) noise??, do not discretize?
		v = v + randn(size(v));
		h = v; % visible state is the previous layer's hidden state
		h = h / alpha;
		h = rand(size(h)) < h; 
	end
	
	X(n,:) = v; % we save the real visible state
end


