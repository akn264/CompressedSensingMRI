# akn264_CSMRI.py
# my matrix-free implementation of ISTA for CS MIR
# thank you

import scipy.io as spio
from scipy import signal
import time
import numpy as np
import matplotlib.pyplot as plt
from project_funs import conv2t, gauss_kernel, fft2c, ifft2c, ista_CSmri

# Loading Data thanks to Miki Lustig of UC Berkeley
mat = spio.loadmat('brain.mat', squeeze_me = True)
# loads it as a dictionary

img = mat['im']
# print(type(img))

# Variable-density sampling matrix things
pdf_vardens = mat['pdf_vardens']
mask_vardens = mat['mask_vardens']

plt.figure(figsize=[10,10])
plt.imshow(abs(img), cmap='gray')
plt.title('Axial T2 Weighted Brain Image... a classic')
plt.axis('off')

# finite difference equation operators 
# h = np.array([[1, 0, -1],[1,0,-1],[1,0,-1]])
# used first order central difference operator, and made into a 3x3. faciliated the convolution operators
h = np.array([[-1/2, 0, 1/2],[-1/2, 0, 1/2],[-1/2, 0, 1/2]])
# h = h.reshape(1,len(h))
hff = np.fliplr(np.flipud(h))
# hf = np.flipud(h)
H = lambda x: signal.convolve2d(x, hff)
Ht = lambda x: conv2t(h, x)

# v = np.array([[1],[0],[-1]])
v = h.T
vff = np.fliplr(np.flipud(v))
V = lambda x: signal.convolve2d(x, vff)
Vt = lambda x: conv2t(v, x)


# stuff for uniform undersampling
m, n = img.shape
G = np.ones((m, n)) / (3/2)
# G = np.ones((m, n)) / 3
G_unif = np.ones((m, n)) / 3
G_unif2 = np.ones((m, n)) / (3/2)
# extract small sample of signal
k = round(m * n * (2/3))  # throwing away 2/3 of the data
ri = np.random.choice(m * n, k, replace=False)
G_unif.flat[ri] = 0
k2 = round(m * n * (1/3)) # throwing away 1/3 of the data
ri2 = np.random.choice(m * n, k2, replace=False)
G_unif2.flat[ri2] = 0
# print(np.sum(G_unif2), np.sum(G_unif),'\n', k, k2)
# plt.figure(figsize=[10,10])
# plt.imshow(G_unif, cmap='gray')
# plt.title('Uniform Undersampling Mask')
# plt.axis('off')


# compare to Gaussian kernel and mask
# plt.figure(figsize=[10,10])
# plt.imshow(mask_vardens, cmap='gray')
fig, [ax1, ax2, ax3] = plt.subplots(1,3, figsize=[10,5])
ax1.imshow(G_unif2, cmap='gray'); ax1.set_title('Uniform Mask 1/3 Undersampled')
ax1.axis('off')
ax2.imshow(G_unif, cmap='gray'); ax2.set_title('Uniform Mask 2/3 Undersampled')
ax2.axis('off')
ax3.imshow(mask_vardens, cmap='gray'); ax3.set_title('Variable-Density Mask 2/3 Undersampled')
ax3.axis('off')


# model the undersampled k-space data
stringg = ['Variable-Density', 'Uniform 1/3', 'Uniform 2/3']
mask = [mask_vardens, G_unif2, G_unif]
pdf = [pdf_vardens, G]
data = np.multiply(fft2c(img), mask[0])
under_dat = ifft2c(np.divide(data, pdf[0]))

# to visualize what undersampling the k-space data does in the image domain
plt.figure(figsize=[10,10])
plt.imshow(np.abs(under_dat), cmap='gray')
plt.title('Undersampled Data with %s mask' % (stringg[0]))
plt.axis('off')

print('\nFirst close these plots.\nThen, the compressed sensing reconstruction will begin.\n')
plt.show()


# let's call ISTA. close plots first!
Nit = 30
alpha = 5
lam = 0.001

print('Running...')
xhat, J = ista_CSmri(data, H, Ht, V, Vt, lam, alpha, Nit)
# plotting occurs in the ista function


time.sleep(10)		# have a lil more time to look at the figures
print('Done!~')