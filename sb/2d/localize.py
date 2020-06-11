# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:38:28 2020
This program carries out the localization on the
2D bundled tubes long sequence
@author: moss
"""
from SparseBayes import SparseBayes
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
np.random.seed(42);

#data directory strings



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
imdir = '/home/moss/SMLM/data/sequence';
start = 10;
def readImages(imageDir,nImages):
    imA = np.zeros((n_grid,n_grid,nImages));
    
    for i in range(0,nImages):
        imageId = str(i+1).zfill(5);
        impath = imageDir + '/' + imageId + '.tif';
        imA[:,:,i] = plt.imread(impath);
        
    return imA;

def runSB(psf,psf_k,imageArray):
        nImages = np.shape(imageArray)[2];
        results = imageArray*0;
        
        for imageIdx in range(0,nImages):
            if imageIdx<=start:
                continue;
            img = imageArray[:,:,imageIdx];
            sub = img-np.average(img[img<np.average(img)+3*np.std(img)]);
            subnorm = sub/np.max(sub);
            sb = SparseBayes(subnorm,psf,psf_k);
            results[:,:,imageIdx] = sb.res;
            s = './'+str(imageIdx+1).zfill(5)+'.out';
            np.savetxt(s,sb.res)
            #plt.imshow(results[:,:,imageIdx]);
            #plt.show();
        return results;


        
nImages = 100;
imageArray = readImages(imdir,nImages);
res = runSB(psf,psf_k,imageArray);
