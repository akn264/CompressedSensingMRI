%% ista_CSmri.m
% Matrix-free implementation of ISTA for CS MRI
function [x, J] = ista_CSmri(y, H, Ht, V, Vt, lam, alpha, Nit)


J = zeros(1, Nit);          % cost (objective) function
x = y;                      % initialize
T = lam / (2 * alpha);

figure()
for k = 1:Nit
    
    Hx = H(x);
    Vx = V(x);
    Fx = fft2c(x);
    Ft = ifft2c(y);
    s = Fx(:) - y(:);
    ss = Hx(:) + Vx(:);
    J(k) = sum(abs(s .^ 2)) + lam * sum(abs(ss));  % cost func
    x = SoftThreshComplex((x + (Ft - (ifft2c(Fx) + lam * Ht(Hx) + lam * Vt(Vx)))/alpha), T);
%     disp(squeeze(Ht(Hx)));
    imshow(abs(x), [0,1])
    title(sprintf('Reconstruction, iteration %d', k))
    drawnow
    pause(0.01)

end

    
    
    
