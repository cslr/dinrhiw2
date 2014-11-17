% extracts single (k:th) image/photo from bigmatrix
% K = 16, px = 13, py = 13
% k = [1:16]

function p = extract_photo(k, photos)

px = 13;
py = 13;
p = zeros(py,px);


for j=1:py
    for i=1:px
        p(j,i) = photos((k-1)*px + i, j);
    end
end


 