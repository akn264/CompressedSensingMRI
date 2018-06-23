function f = conv2t(h,g)
% f = convt(h, g);
% Transpose convolution f = H'g
[m, n] = size(h);
[mg, ng] = size(g);
f = conv2(flip(flip(h(m:-1:1,n:-1:1),2),1), g);
f = f(m:mg,n:ng);
