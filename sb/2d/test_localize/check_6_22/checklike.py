# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:55:26 2020

@author: moss
"""

import math as math
import autograd as Agrad
import autograd.numpy as np 
import autograd.numpy.fft as fft
#import numpy as np
#import numpy.fft as fft
import scipy.optimize
import scipy.stats as st
import scipy.signal as sg
from scipy.integrate import trapz
from scipy.integrate import simps
from photutils import find_peaks
from photutils import detect_threshold
# -- plotting --- 
import matplotlib as mpl 
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False
mpl.rcParams.update({'font.size': 22})

#global variables about data
n_grid = 64; #pix
pix_1d = np.linspace(0., 1., n_grid) # pixel gridding

pix_scale = 100; #nm per pixel
back_std = 22.0; #given by challenge, may want to est from image w/ 3sig clip
lam = 723.0; #wavelength
NA = 1.4 #numerical aperture;
FWHM = lam/(2*NA); #fwhm in nm of gaussian psf
sig_nm = FWHM/(2*np.log(2.0));
sig_psf = sig_nm/100/64;#gaussian sigma in pix
sig_sq = sig_psf**2 #so we don't have to compute

sig_noise = back_std;

#create our psf
mid = int(n_grid/2);
x,y = np.meshgrid(pix_1d,pix_1d);
psf = np.exp(-((y-pix_1d[mid])**2 + (x - pix_1d[mid])**2)/2/sig_psf**2); #keep in mind difference between x and y position and indices! Here, you are given indices, but meshgrid is in x-y coords
#fourier transform of psf
psf_k = fft.fft2(psf);
img = plt.imread('/home/moss/SMLM/data/sequence/00002.tif');
img = img-np.average(img[img<np.average(img)+3*np.std(img)]);
data = img/np.max(img);
xi = data + 0.5;
f = 7/(64**2);
sig_noise = np.std(data[data<np.average(data)+3*np.std(data)]);
print('noise is');
print(sig_noise);
norm_sig = 0.75;
norm_mean = -5;
wlim = (0.01,5);
def roll_fft(f):
    r,c = np.shape(f);
    f2 = np.roll(f,(c//2));
    f3 = np.roll(f2,(r//2),axis=-2);
    return f3;
def lognorm(ws):
    return np.exp(-0.5*(np.log(ws) - norm_mean)**2 /norm_sig**2)/np.sqrt(2*np.pi)/norm_sig/ws;
def loss_like(ws_k):
    #gaussian likelihood, assumes ws_k is in complex form and 2d
    conv = np.real(fft.ifft2(ws_k*psf_k)); #convolution of ws with psf
    like_loss = 0.5 * (conv - data)**2 /sig_noise**2 #gaussian likelihood loss
    
    return like_loss;
def loss_prior(ws,f,alpha):
    #prior, assumes ws is real and 2D
    print(ws);
    w_norm = (wlim[1]**(alpha+1) - wlim[0]**(alpha+1))/(alpha+1); #normalization from integrating
    p1 = ws**alpha /w_norm;
    prior = np.where(ws<=0.,0.,np.log(lognorm(ws)*(1-f) + f*p1))
    return prior;

def loss_fn(wsp_k,xi,f,alpha):
    wsp = np.real(fft.ifft2(wsp_k));
    ws = xi*np.log(np.exp(wsp/xi)+1) #reparametrize from m_prime back to m
    ws_k = fft.fft2(ws);
    return loss_like(ws_k) - loss_prior(ws,f,alpha);
    
m = np.loadtxt('./00002.out')
m_k = fft.fft2(m);

like = loss_like(m_k);
prior = loss_prior(m,f,-1.25);

np.savetxt('like.txt',roll_fft(like));
np.savetxt('prior.txt',roll_fft(prior));
np.savetxt('loss.txt',roll_fft(like-prior));
np.savetxt('rollres.txt',roll_fft(m));

fig, ax = plt.subplots(1,3)
ax[0].imshow(roll_fft(m));
ax[0].set_title('Sparse Bayes')
ax[1].imshow((like));
ax[1].set_title('Like')
ax[2].imshow(roll_fft(prior));
ax[2].set_title('Prior')

plt.show();

