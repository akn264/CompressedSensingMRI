# A collection of functions for EL6123 Final Project
# project_funs.py
# Aimee Nogoy akn264 
# NYU EE BS/MS 2018 WE OUT!!! LOL why did I say this
# thank you!!!

import numpy as np
from scipy import signal, fftpack
import matplotlib.pyplot as plt


def softComplex(x, T):
	# out = (abs(x) > lambda) .* (x .* (abs(x) - lambda) ./ abs(x))
	absx = np.absolute(x)
	arg1 = (absx > T)
	arg2 = np.multiply(x, (absx - T))
	arg22 = np.divide(arg2, absx)
	out = np.multiply(arg1, arg22)

	return(out)

# inspired by Prof. Ivan Selesnick's convt MATLAB function.
def conv2t(h, g):
	mh, nh = h.shape
	mg, ng = g.shape
	hff = np.fliplr(np.flipud(h))
	# print('error lmao sorry fam')
	# hf = np.fliplr(h)
	f = signal.convolve2d(g, hff[::-1])
	# print('error lmao sorry fam')
	f = f[mh:mg+1, nh:ng+1]
	# print(f.shape)

	return(f)

def gauss_kernel(nx, ny, sig):
	# from sdrangan
	# creates the Gaussian kernel of size (nx, ny) with standard dev sig
	dxsq = (np.arange(nx) - (nx - 1) / 2) ** 2
	dysq = (np.arange(ny) - (ny - 1) / 2) ** 2
	dsq = dxsq[:,None] + dysq[None,:]
	G = np.exp(-0.5 * dsq / (sig ** 2))
	G = G / np.sum(G)

	return(G)

# the next two FT functions are orthonormal and centered. adapted from Lustig's MATLAB functions
def ifft2c(x):
	out = np.sqrt(len(x.ravel())) * fftpack.ifftshift(fftpack.ifft2(fftpack.fftshift(x)))
	return(out)

def fft2c(x):
	out = (1 / np.sqrt(len(x.ravel()))) * fftpack.fftshift(fftpack.fft2(fftpack.ifftshift(x)))
	return(out)


def ista_CSmri(y, H, Ht, V, Vt, lam, alpha, Nit):
	plt.ion()
	fig, [ax1,ax2] = plt.subplots(1,2, figsize=[16,8])

	J = np.zeros(Nit)
	x = y
	T = lam / (2 * alpha)

	for k in range(Nit):
		Hx = H(x)
		Vx = V(x)
		# print(Hx.shape, Vx.shape)
		Fx = fft2c(x)
		Ft = ifft2c(y)
		# print(Fx.shape, Ft.shape)
		s = Fx.ravel() - y.ravel()
		ss = Hx.ravel() + Vx.ravel()
		J[k] = np.sum(np.abs(s ** 2)) + lam * np.sum(np.abs(ss))
		x = softComplex((x + (Ft - (ifft2c(Fx) + lam * Ht(Hx) + lam * Vt(Vx))) / alpha), T)

		
		ax1.imshow(abs(x), cmap='gray', vmin=0, vmax=1)
		ax2.plot(k, J[k], color = 'red', marker = 'o', ms=10)
		
		ax1.set_title('Reconstruction - Iteration {0}'.format(k))
		ax1.axis('off')

		ax2.set_title('Cost Function - Iteration {0}'.format(k))

		plt.pause(0.01)
		plt.draw()		# please enjoy the real time plotting update
		plt.show()

	print('Done here!~')

	return(x, J)

# didn't use this here
def ista_conv(y, H, Ht, alpha, lam, Nit):
	J = np.zeros((1, Nit))
	x = 0 * Ht(y)
	# print(len(Ht(y)))
	# print(x.shape)
	T = lam / (2 * alpha)

	J = np.array([])

	for k in range(Nit):
		Hx = H(x)

		s = Hx.ravel() - y.ravel()
		ss = x.ravel()
		j = np.sum(np.abs(s ** 2)) + lam * np.sum(np.abs(ss))
		J = np.append(J, j)
		x = softComplex((x + (Ht(y - Hx)) / alpha), T)
		# print('size of x:', x.shape)
		# print('new x:', x)
		# print('J is: ', J)

	return x, J
