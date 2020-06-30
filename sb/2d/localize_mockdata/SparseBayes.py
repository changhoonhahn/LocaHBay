# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:51:52 2020
Class for implementing the sparse bayes optimization
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


class SparseBayes:
    
    #constructor
    def __init__(self,data,psf,psf_k,no_source):
        self.data = np.absolute(data);
        self.psf = psf;
        self.psf_k = psf_k;
        self.n_grid = len(data);
        self.f_true = no_source/self.n_grid**2
        self.wlim = (1,2); #min and max signal (given by challenge, will determine ourselves in the future)
        self.sig_noise = self.getNoise()
        self.norm_mean, self.norm_sig = self.getNorm();
        self.xi = self.getXi();
        self.res = self.run();
        
    #In order to minmize w.r.t. complex numbers we embed the complex nummbers into real space with twice the dimensions
    #e.g. C -> 2R
    #here are our helper functions for this
    def real_to_complex(self,z):      # real vector of length 2n -> complex of length n
        return z[:len(z)//2] + 1j * z[len(z)//2:]
    
    def complex_to_real(self,z):      # complex vector of length n -> real of length 2n
        return np.real(np.concatenate((np.real(z), np.imag(z))));
    
    
    def getNoise(self):
        #estimate based off of vals less than 3 sigma above mean
        #did a couple of spot checks and this works pretty well, need more rigor in future
        noi = np.std(self.data[self.data<np.average(self.data)+2*np.std(self.data)]);
        print('noise is:');
        print(noi);
        return noi;
         
    def getXi(self):
        return self.data+0.7;
    
    def getNorm(self):
        #return (np.log(np.average(data[data<np.average(data)+3*np.std(data)])-self.sig_noise),np.average(data)*0.5); #really just assigning random values here, need better way to do this
        #return (np.log(0.1*self.sig_noise),np.average(data)*0.5);
        #print(np.average(data)*5.)
        #return (-1,np.average(data)*5.);
        return (-5,1);
    #expects ws, NOT the fourier coefficients ws_k
    def lognorm(self,ws):
        return np.exp(-0.5*(np.log(ws) - self.norm_mean)**2 /self.norm_sig**2)/np.sqrt(2*np.pi)/self.norm_sig/ws;
    
    #derivative of lognorm
    def diff_lognorm(self,ws):
        #taken from: https://iopscience.iop.org/article/10.1088/1742-6596/1338/1/012036/pdf
        df = -1*self.lognorm(ws)*(1/ws - (np.log(ws)-self.norm_mean)/ws/self.norm_sig**2);
        
        return df;
          
    #now we define our loss function (basically log likelihood)
    #ws is the fourier coefficients embedded into the reals and flattened to a 1d array
    
    def loss_like(self,ws_k):
        #gaussian likelihood, assumes ws_k is in complex form and 2d
        conv = np.real(fft.ifft2(ws_k*self.psf_k)); #convolution of ws with psf
        like_loss = 0.5 * np.sum((conv - self.data)**2) /self.sig_noise**2 #gaussian likelihood loss
        
        return like_loss;
    def loss_prior(self,ws,f,alpha):
        #prior, assumes ws is real and 2D
        print(ws);
        #w_norm = (self.wlim[1]**(alpha+1) - self.wlim[0]**(alpha+1))/(alpha+1); #normalization from integrating
        w_norm = np.sum(np.linspace(self.wlim[0],self.wlim[1],1000)**alpha);
        p1 = ws**alpha /w_norm;
        prior = np.where(ws<=0.,0.,np.log(self.lognorm(ws)*(1-f) + f*p1))
        prior_loss = np.sum(prior);
        return prior_loss;
    
    def loss_fn_real(self,wsp_k,xi,f,alpha):
        wsp_k = wsp_k.reshape((self.n_grid,self.n_grid)); #reshape to 2d
        wsp = np.real(fft.ifft2(wsp_k));
        ws = xi*np.log(np.exp(wsp/xi)+1) #reparametrize from m_prime back to m
        ws_k = fft.fft2(ws);
        return self.loss_like(ws_k)-self.loss_prior(ws,f,alpha);
    
    def loss_fn(self,wsp_k,xi,f,alpha):
        wsp_k = self.real_to_complex(wsp_k); #2*reals -> complex
        wsp_k = wsp_k.reshape((self.n_grid,self.n_grid)); #reshape to 2d
        wsp = np.real(fft.ifft2(wsp_k));
        ws = xi*np.log(np.exp(wsp/xi)+1) #reparametrize from m_prime back to m
        ws_k = fft.fft2(ws);
        return self.loss_like(ws_k) - self.loss_prior(ws,f,alpha);
        
    #numerical gradient (may run faster)    
    def areal(self,th,xi,f,alpha):
        th_comp = self.real_to_complex(th);
        gfunc = Agrad.holomorphic_grad(lambda tt: self.loss_fn_real(tt,xi,f,alpha));
        grad_comp = gfunc(th_comp);
        grad_real = self.complex_to_real(np.conj(grad_comp));
        return grad_real;
    
    #analytical gradient or likelihood
    def grad_like(self,wsp,ws,ws_k,xi):
        #print('start grad_like')
        conv = np.real(fft.ifft2(ws_k*self.psf_k)); #convolution of ws with psf
        term1 = (conv - self.data)/self.n_grid**2 /self.sig_noise**2 #term thats squared in like (with N in denom)
        grad = np.zeros((self.n_grid,self.n_grid),dtype='complex')
        for i in range(0,self.n_grid):
            for j in range(0,self.n_grid):        
                #try to modulate by hand
                ft1 = fft.fft2(1/(1+np.exp(-1*wsp/xi)));
                ftp = np.roll(ft1,(i,j),axis=(0,1));
                term2 = fft.ifft2(ftp*self.psf_k);
                grad[i,j] = np.sum(term1*term2);
        grad_real = self.complex_to_real(np.conj(grad.flatten())); #embed to 2R
        #print('end grad_like');
        return grad_real; #return 1d array
    
    #analytical gradient of prior
    def grad_prior(self,wsp,ws,ws_k,xi,f,alpha):
        w_norm = (self.wlim[1]**(alpha+1) - self.wlim[0]**(alpha+1))/(alpha+1); #normalization from integrating
        param_term = 1/(1+np.exp(-1*wsp/xi)) #differentiation term due to parametrization
        #grad = fft.ifft2((-1/ws - (np.log(ws)-norm_mean)/ws/norm_sig**2)*param_term); #version with p1=0
        numerator = (1+(np.log(ws)-self.norm_mean)/self.norm_sig**2)*self.lognorm(ws)/ws + f*alpha*ws**(alpha-1)/w_norm;
        prior = self.lognorm(ws)*(1-f) + f*ws**(alpha)/w_norm; #prior w/o log
        grad = fft.ifft2(param_term*numerator/prior);
        grad_real = self.complex_to_real(np.conj(grad.flatten())); #embed to 2R
        return grad_real; #return 1d array
    
    def grad_loss(self,wsp_k,xi,f,alpha):
        wsp_k = self.real_to_complex(wsp_k); #2*reals -> complex
        wsp_k = wsp_k.reshape((self.n_grid,self.n_grid)); #reshape to 2d
        wsp = np.real(fft.ifft2(wsp_k));
        ws = xi*np.log(np.exp(wsp/xi)+1) #reparametrize from m_prime back to m
        ws_k = fft.fft2(ws);
        
        return self.grad_like(wsp,ws,ws_k,xi)-self.grad_prior(wsp,ws,ws_k,xi,f,alpha);    
    
    def optimize_m(self,wsp_k,xi,f,alpha):
        print('optimizing')
        gradfun = lambda tg: self.areal(tg,xi,f,alpha);
        res = scipy.optimize.minimize(lambda tt: self.loss_fn(tt,xi,f,alpha),
            wsp_k, # theta initial
            jac=gradfun,                          
            method='Newton-CG');
        print('Number of Iterations m');
        print(res['nit']);
        w_final_k = self.real_to_complex(res['x']);
        w_final_k = w_final_k.reshape((self.n_grid,self.n_grid)); #reshape to 2d
        w_final = np.real(fft.ifft2(w_final_k));
        w_final = xi*np.log(np.exp(w_final/xi)+1);
        return w_final;
        
        
    def run(self):
        #plt.imshow(self.data);
        #plt.show();
        #plt.imshow(self.psf);
        #plt.show();
        #set up initial guesses
        #create initial parameters
        tt0 = np.zeros((self.n_grid,self.n_grid)) + 2*self.wlim[1]; #begin with high uniform mass in each pixel
        tt0 = self.xi*np.log(np.exp(tt0/self.xi)-1);
        tt0_k = fft.fft2(tt0); #take fft
        t_ini = self.complex_to_real(tt0_k.flatten()) #flatten to 1d for scipy and embed in 2R
        
        alpha_ini = -1.25;
        f_ini = self.f_true;
        return self.optimize_m(t_ini,self.xi,f_ini,alpha_ini);