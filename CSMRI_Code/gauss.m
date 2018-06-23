function [H, Hf] = gauss(sig, nx, m, n)
% generate 2d Gaussian Filter for the variable-density probability
% distribution

x = [-nx:1.0:nx];
gauss = exp(-x.^2 / (2*sig^2));
gauss2 = gauss' * gauss;
gauss2 = gauss2 / (sum(sum(gauss2)));
H = gauss2;
Hf = freqz2(gauss2, [m n]);