% householder vector calculation

function v = teu_house(x)
n = length(x);
u = norm(x);
v = x;

if u ~= 0
    % b = x(1) + sign(x(1)) * u; % real
    % v([2:n]) = v([2:n])/b;
    
    if(isreal(x(1)))
      v(1) = v(1) + sign(x(1))*u;
    else
      t = v(1) + u * exp(angle(x(1))*i); % complex
      s = v(1) - u * exp(angle(x(1))*i); % complex
    
      if(abs(t) > abs(s))
        v(1) = t;
      else
        v(1) = s;
      end
    end
    
    v = v / v(1);
end

v(1) = 1;
