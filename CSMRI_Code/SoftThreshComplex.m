function [out] = SoftThreshComplex(x, lambda)
% element wise
% out = zeros(1, length(x));
% for i = 1:length(x)
%     if abs(x(i)) <= lam
%         out(i) = 0;
%     elseif abs(x(i)) > lam
%         out(i) = x(i) * (abs(x(i)) - lam) / abs(x(i));
%     end
% end
% 
% end

out = (abs(x) > lambda).*(x.*(abs(x)-lambda)./(abs(x)+eps));
    