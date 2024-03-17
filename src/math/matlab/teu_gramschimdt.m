
% calculates gram-schmidt orthogonalization 
% for a given (partial) basis {B(:,1),B(:,2)..} and then normalizes vectors

function U = teu_gramschimdt(B)

[N,M] = size(B);
Z = zeros(N,1);


for i=1:M
    
    Z = zeros(N,1);
    for j=1:(i-1)
        Z = Z + (U(:,j)' * B(:,i)) * U(:,j);
    end
    
    U(:,i) = B(:,i) - Z;
    U(:,i) = U(:,i) / norm(U(:,i));
end


