% CSMRI_Project_DSP2.m
% EL7133 Final Project
% Aimee Nogoy and Anthony Mekhanik


%%
clear; clc; close all
%%
% load the data
load brain.mat
% mat file contains the image data, and Gaussian sensing matrix and mask
%%
figure(1), imshow(abs(im),[])
title('Axial T2 Weighted Brain image...a classic');

%% first-order finite difference operators
% try second or third order

% h = [1 -1];
h = [1 0 -1; 2 0 -2; 1 0 -1];
% h = [1 2 -1];
H = @(x) conv2(flip(flip(h, 2), 1), x);
Ht = @(x) conv2t(h, x);

% v = [1; -1];
% v = h';
v = [1 2 1; 0 0 0; -1 -2 -1];
V = @(x) conv2(flip(flip(v, 2), 1), x);
Vt = @(x) conv2t(v, x);

%% compute mask and pdf
% own pdf that was used for comparison
sig = 2;
nx = 15;   % should be > 5*sigma, odd
[m, n] = size(im);
[~, pdf] = gauss(sig, nx, m, n);
% figure
% imshow(pdf, [])
% figure
% mesh(pdf); colorbar

%% linear sampling
%  was used for comparison. 
k = round(m * n * (3/4));     
ri = randperm(m * n, k);
pdff = pdf(:);
pdff(ri) = 0; 
mask = logical(reshape(pdff,[512,512]));
figure
imshow(mask, [])


%%
% DATA = fft2c(im) .* mask_vardens;
% im_cs = ifft2c(DATA ./ pdf_vardens);

DATA = fft2c(im) .* mask_unif;
im_cs = ifft2c(DATA ./ pdf_unif);
figure
imshow(abs(im_cs),[])
title('undersampled data')

%%
% figure(3);

Nit = 50;
alpha = 5;
lam = 0.001;

[xhat, J] = ista_CSmri(DATA, H, Ht, V, Vt, lam, alpha, Nit);

figure
plot(J)
title('Cost Function')


%%
% format long e
err = immse(im, xhat);
fprintf('\nThe mean-squared error is %0.9f\n', err);
