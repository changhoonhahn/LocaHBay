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
from astropy.table import Table
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
n_grid = 65; #pix
pix_1d = np.linspace(0., 1., n_grid) # pixel gridding

sig_psf = 0.03;#gaussian sigma in pix
print('sig_psf');
print(sig_psf);
sig_sq = sig_psf**2 #so we don't have to compute


#create our psf
mid = int(n_grid/2);
x,y = np.meshgrid(pix_1d,pix_1d);
psf = np.exp(-((y-pix_1d[mid])**2 + (x - pix_1d[mid])**2)/2/sig_psf**2); #keep in mind difference between x and y position and indices! Here, you are given indices, but meshgrid is in x-y coords
#fourier transform of psf
psf_k = fft.fft2(psf);
start = 0;
def readImages(imageDir,nImages):
    imA = np.zeros((n_grid,n_grid,nImages));
    
    for i in range(0,nImages):
        imageId = str(i)
        impath = imageDir + str(imageId) + '.dat'
        imA[:,:,i] =np.loadtxt(impath);
        
    return imA;
def runSB(imageDir,saveDir,psf,psf_k,imageArray):
        nImages = np.shape(imageArray)[2];
        results = imageArray*0;
        
        for imageIdx in range(0,nImages):
            if imageIdx<start:
                continue;
            grndpath = imageDir + str(imageIdx) + '.truth';
            grnd = np.loadtxt(grndpath);
            no_source = np.shape(grnd)[0];
            if len(np.shape(grnd)) <2:
                no_source = 1;
            img = imageArray[:,:,imageIdx];
            sb = SparseBayes(img,psf,psf_k,no_source);
            results[:,:,imageIdx] = sb.res;
            s = saveDir+str(imageIdx)+'.out';
            np.savetxt(s,sb.res)
            #plt.imshow(results[:,:,imageIdx]);
            #plt.show();
        return results;


        
nImages = 25;
datasets = ['hd_lsnr']
for d in datasets:
    imdir = '/home/moss/Projects/LocaHBay/sb/2d/data_alpha/' + d + '/';
    svdir = '/home/moss/Projects/LocaHBay/sb/2d/localize_mockdata/' + d +'/';
    imageArray = readImages(imdir,nImages);
    res = runSB(imdir,svdir,psf,psf_k,imageArray);
