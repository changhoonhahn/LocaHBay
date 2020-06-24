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


class SparseBayes_nofft:
    
    #constructor
    def __init__(self,data,psf,psf_k,sig_psf,no_source):
        self.sig_psf = sig_psf
        self.data = data;
        self.psf = psf;
        self.psf_k = psf_k;
        self.n_grid = len(data);
        self.f_true = no_source/self.n_grid**2
        self.pix_1d = np.linspace(0., 1., len(data)) # pixel gridding
        self.wlim = (0.01,5000); #min and max signal (given by challenge, will determine ourselves in the future)
        self.sig_noise = self.getNoise(data)
        self.norm_mean, self.norm_sig = self.getNorm(data);
        self.xi = self.getXi(data);
        self.res = self.run();
        
    
    
    
    def getNoise(self,data):
        #estimate based off of vals less than 3 sigma above mean
        #did a couple of spot checks and this works pretty well, need more rigor in future
        return np.std(data[data<np.average(data)+3*np.std(data)]);
         
    def getXi(self,data):
        return data+0.5;
    
    def getNorm(self,data):
        #return (np.log(np.average(data[data<np.average(data)+3*np.std(data)])-self.sig_noise),np.average(data)*0.5); #really just assigning random values here, need better way to do this
        #return (np.log(0.1*self.sig_noise),np.average(data)*0.5);
        #print(np.average(data)*5.)
        return (-1.,.75);
    def gaussian(self,x, loc=None, scale=None): 
        
        y = (x - loc)/scale
        return np.exp(-0.5*y**2)/np.sqrt(2.*np.pi)/scale

    def psi(self,pos): 
        ''' measurement model, which in our case is just a 1d gaussian of width 
        sigma (PSF) written out to a meshgrid created by pix1d 
        '''
        x,y = np.meshgrid(self.pix_1d,self.pix_1d);
        return np.exp(-((y-self.pix_1d[pos[0]])**2 + (x - self.pix_1d[pos[1]])**2)/2/self.sig_psf**2); #keep in mind difference between x and y position and indices! Here, you are given indices, but meshgrid is in x-y coords
        
    def Psi(self,ws): 
        ''' "forward operator" i.e. forward model 
        
        Psi = int psi(theta) dmu(theta) 
    
        where mu is the signal parameter
        '''
        return np.sum(np.array([w*self.psi(index) for (index,w) in np.ndenumerate(ws)]),0)          
        #now we define our loss function (basically log likelihood)
    def prior_i(self,w,fdensity,alpha,sig):
        '''
        log of Poisson prior for an indivudial pixel
        '''
        w_norm = (self.wlim[1]**(alpha+1) - self.wlim[0]**(alpha+1))/(alpha+1); #normalization from integrating
        pri=0.;
        if 0. < w :
            #norm = (max(interval_grid)**(alpha+1) - min(interval_grid)**(alpha+1))/(alpha+1); #normalization of mass function
            p1 =0;# w**alpha /w_norm; #probability of single source
            #w_fft = np.linspace(0,w,50);
            #now probability of second source
            #p2 = np.abs(Agrad.numpy.fft.ifft(Agrad.numpy.fft.fft(w_fft**alpha /w_norm)**2));
            #p2 = p2[-1];
            pri += fdensity*p1 #+ p2*fdensity**2
        if w > 0:
            #pri += (1.-fdensity - fdensity**2 ) *gaussian(np.log(w),loc=-4., scale=sig)/w
            pri += (1.-fdensity) *self.gaussian(np.log(w),loc=self.norm_mean, scale=sig)/w
        return pri
        
    def lnprior(self,ws,fdensity,alpha,sig): 
    	'''
    	calculate log of prior
    	'''
    	return np.sum([np.log(self.prior_i(w,fdensity,alpha,sig)) for w in ws.flatten()])
    
    def lnlike(self,ws): 
        ''' log likelihood 
        '''
        return -0.5 * np.sum((self.Psi(ws) - self.data)**2/self.sig_noise**2)
     
    def lnpost(self,ws,fdensity,alpha,sig): 
        #converting flattened ws to matrix
        ws = ws.reshape((self.n_grid,self.n_grid));
        post = self.lnlike(ws) + self.lnprior(ws,fdensity,alpha,sig);
        return post;
    
    def grad_lnpost(self,ws,fdensity,alpha,sig):
        #calculate gradient of the ln posterior
        print('grad');
        w_norm = (self.wlim[1]**(alpha+1) - self.wlim[0]**(alpha+1))/(alpha+1); #normalization from integrating
        mo = np.exp(self.norm_mean);
        ws = ws.reshape((self.n_grid,self.n_grid));
        #calc l1
        bsis = (self.Psi(ws)-self.data)/self.sig_noise**2;
        lsis = ws*0;
        for (index,w) in np.ndenumerate(ws):
            lsis[index] = np.sum(self.psi(index)*bsis);
        l1 = lsis#*np.sum((Psi(ws)-data)/2/sig_noise**2);
        xsi = (1.-fdensity ) * self.gaussian(np.log(ws),loc=np.log(mo), scale=sig)/ws + fdensity*(ws**alpha /w_norm)
        l2 = -1*self.gaussian(np.log(ws),loc=np.log(mo), scale=sig)*(1.-fdensity)/ws**2 - (1.-fdensity)*np.log(ws/mo)*np.exp(-np.log(ws/mo)**2 /2/sig**2)/np.sqrt(2*np.pi)/ws**2 /sig**3 + fdensity*alpha*ws**(alpha-1) /w_norm;
        l2 = l2/np.absolute(xsi);
        l_tot = l1-l2;
        return l_tot.flatten();   

        
    def optimize_m(self,t_ini, f_ini,alpha_ini, sig_curr):
        #keeping in mind that minimize requires flattened arrays
        afunc = Agrad.grad(lambda tt: -1*self.lnpost(tt, f_ini,alpha_ini, sig_curr));
        grad_fun = lambda tg: self.grad_lnpost(tg,f_ini,alpha_ini,sig_curr);
        #hfunc = Agrad.hessian(lambda tt: -1*lnpost(tt, f_ini,alpha_ini, sig_curr));
        hess_fun = lambda th: self.hess_lnpost(th,f_ini,alpha_ini,sig_curr);
        #grad_fun = Agrad.grad(lambda tg: -1*lnpost(tg,f_ini,alpha_ini,sig_curr));
        
        res = scipy.optimize.minimize(lambda tt: -1*self.lnpost(tt,f_ini,alpha_ini,sig_curr),
                                      t_ini, # theta initial
                                      jac=grad_fun, 
                                      method='L-BFGS-B', 
                                      bounds=[(1e-5, 1000)]*len(t_ini))
                                      
                                                                
        tt_prime = res['x'];
        print('Number of Iterations:')
        print(res['nit'])
        print('Final Log-Likelihood:')
        w_final = tt_prime.reshape((self.n_grid,self.n_grid));
        print(-1*self.lnpost(w_final,f_ini,alpha_ini,sig_curr));
        return w_final;        
    def run(self):
        plt.imshow(self.data);
        plt.show();
        #plt.imshow(self.psf);
        #plt.show();
        #set up initial guesses
        #create initial parameters
        t_ini = np.zeros((self.n_grid,self.n_grid)).flatten() + self.wlim[1]; #begin with high uniform mass in each pixel
        
        alpha_ini = 3.;
        f_ini = self.f_true
        return self.optimize_m(t_ini,f_ini,alpha_ini,self.norm_sig);