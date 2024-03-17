% generates data from the stacked RBMs using hidden states DATA (for the top layer)

function X = stackedRBM_generate_data(DATA, stack)

% we generate random points for the top layer of the RBM
% and then generate the visible layer from that state

DEEPNESS = length(stack);

NUMDATA = size(DATA, 1);

% random [0,5] hidden states of the top layer
% DATA = rand(NUMDATA, length(stack{DEEPNESS}.rbm.b));
DATA = rand(size(DATA)) < DATA; % DISCRETIZES THEM

for n=1:NUMDATA
	h = DATA(n, :)';
	
	% now we backward iterate back to 
	% the visible layer from the top layer
	for d=DEEPNESS:-1:1
		rbm = stack{d}.rbm;
		
		if(rbm.type == 'GB')
		  v = ((rbm.C**-0.5)*rbm.W*h + rbm.a);
		  nn = randn(size(v));
		  [V, L] = eig(rbm.C);
		  nn = V*(L**0.5)*nn;
		  v = v + nn; % adds N(0,C) noise term to G-B network
		else
		  v = (rbm.W*h + rbm.a);
		  v = rand(size(v)) < v; % discretizes B-B network
		end
			
		h = v; % visible state is the previous layer's hidden state 
	end
	
	X(n,:) = v; % we save the real visible state
end


