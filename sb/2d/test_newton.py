
import math as math
import autograd as Agrad
import autograd.numpy as np 
#import numpy as np
import autograd.numpy.fft as fft
import scipy.optimize
import scipy.stats as st
import autograd.scipy.signal as signal
import scipy.signal as scsig
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


########################################################################
'''
rp_2d.py
Code modified from rp_2d.py
Author: Massimo Pascale
Last Updated: 11/12/2019

Code uses poisson prior and exponential intensity function to determine
point source locations in psf+noise and recover hyperparameters.
'''
########################################################################




#create global definitions - this will become a main function later on
np.random.seed(42)
Ndata = 2;
n_grid = 7;
pix_1d = np.linspace(0., 1., n_grid) # pixel gridding
fdensity_true = float(Ndata)/float(n_grid**2); #number density of obj in 1d

sig_psf = 0.1 # psf width
sig_noise = 0.02 # noise level

#these are values for the power law function for sampling intensities
w_interval = (1,2);
w_lin = np.linspace(1,2,100);
alpha_true = 2;
w_norm = (50**(alpha_true+1) - w_interval[0]**(alpha_true+1))/(alpha_true+1);
w_func = np.power(w_lin,alpha_true)/w_norm;
w_true = w_norm*np.random.choice(w_func,Ndata);

mid = int(n_grid/2);
x,y = np.meshgrid(pix_1d,pix_1d);
psf = np.exp(-((y-pix_1d[mid])**2 + (x - pix_1d[mid])**2)/2/sig_psf**2); #keep in mind difference between x and y position and indices! Here, you are given indices, but meshgrid is in x-y coords



def psi(pos): 
    ''' measurement model, which in our case is just a 1d gaussian of width 
    sigma (PSF) written out to a meshgrid created by pix1d 
    '''
    x,y = np.meshgrid(pix_1d,pix_1d);
    return np.exp(-((y-pix_1d[pos[0]])**2 + (x - pix_1d[pos[1]])**2)/2/sig_psf**2); #keep in mind difference between x and y position and indices! Here, you are given indices, but meshgrid is in x-y coords

def gaussian(x, loc=None, scale=None): 
    '''
    scipy's gaussian pdf didn't work idk
    '''
    y = (x - loc)/scale
    return np.exp(-0.5*y**2)/np.sqrt(2.*np.pi)/scale
    
def conv2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape = s, strides = a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM);

def fft_convolve2d(x,y):
    """ 2D convolution, using FFT"""
    fr = fft.fft2(x)
    fr2 = fft.fft2(np.flipud(np.fliplr(y)))
    m,n = fr.shape
    cc = np.real(fft.ifft2(fr*fr2));
    cc = np.roll(cc, -int(m/2)+1,axis=0)
    cc = np.roll(cc, -int(n/2)+1,axis=1)
    return cc
    
def real_to_complex(z):      # real vector of length 2n -> complex of length n
    return z[:len(z)//2] + 1j * z[len(z)//2:]

def complex_to_real(z):      # complex vector of length n -> real of length 2n
    return np.real(np.concatenate((np.real(z), np.imag(z))));
    
def symmetrize(a):
    """
    Return a symmetrized version of NumPy array a.

    Values 0 are replaced by the array value at the symmetric
    position (with respect to the diagonal), i.e. if a_ij = 0,
    then the returned array a' is such that a'_ij = a_ji.

    Diagonal values are left untouched.

    a -- square NumPy array, such that a_ij = 0 or a_ji = 0, 
    for i != j.
    taken from a stack exchange post:
    https://stackoverflow.com/questions/2572916/numpy-smart-symmetric-matrix
    """
    return a + a.T - np.diag(a.diagonal());
    
def convolvesame(m,n):
    size = len(m);
    con = signal.convolve(m,n,'fft');
    pad = int((len(con) - size)/2);

    same = con[pad:len(con)-pad,pad:len(con)-pad];
    return same;
    
def convolvesame_fft(m,n):
    size = len(m);
    con = scsig.convolve2d(m,n,'same','wrap');
    #pad = int((len(con) - size)/2);

    #same = con[pad:len(con)-pad,pad:len(con)-pad];
    return con;
        
def Psi(ws): 
    ''' "forward operator" i.e. forward model 
    
    Psi = int psi(theta) dmu(theta) 

    where mu is the signal parameter
    '''
    return np.sum(np.array([w*psi(index) for (index,w) in np.ndenumerate(ws)]),0)

def prior_i(w,fdensity,alpha,sig):
    '''
    log of Poisson prior for an indivudial pixel
    '''
    #if w <1e-5 or  10 < w:
    #    return np.inf;
    pri=0.;
    if 0. < w <= 4:
        #norm = (max(interval_grid)**(alpha+1) - min(interval_grid)**(alpha+1))/(alpha+1); #normalization of mass function
        p1 = w**alpha /w_norm; #probability of single source
        #w_fft = np.linspace(0,w,50);
        #now probability of second source
        #p2 = np.abs(Agrad.numpy.fft.ifft(Agrad.numpy.fft.fft(w_fft**alpha /w_norm)**2));
        #p2 = p2[-1];
        pri += fdensity*p1 #+ p2*fdensity**2
    if w > 0:
        #pri += (1.-fdensity - fdensity**2 ) *gaussian(np.log(w),loc=-4., scale=sig)/w
        pri += (1.-fdensity) *gaussian(np.log(w),loc=-4., scale=sig)/w
    return pri
    
def lnprior(ws,fdensity,alpha,sig): 
	'''
	calculate log of prior
	'''
	return np.sum([np.log(prior_i(w,fdensity,alpha,sig)) for w in ws.flatten()])

def lnlike(ws): 
    ''' log likelihood 
    '''
    #other = -0.5* np.sum((np.convolve(ws,psf)-data)**2 /sig_noise**2);
    return -0.5 * np.sum((Psi(ws) - data)**2/sig_noise**2)
    
def lnprior_k(ws,fdensity,alpha,sig): 
    #ws = np.absolute(fft.ifft(ws));
    #ws = ws.reshape((n_grid,n_grid));
    pri =  np.sum([np.log(prior_i(w,fdensity,alpha,sig)) for w in ws.flatten()]);
    return pri.flatten();

def lnlike_k(ws): 
    ''' log likelihood 
    '''
    #ws = ws.reshape((n_grid,n_grid));
    #ws = np.absolute(fft.ifft2(ws));
    like = -0.5 * np.sum((fft.ifft2(fft.fft2(ws)*fft.fft2(psf)) - data)**2/sig_noise**2);
    like = np.real(like);
    #print('like is:');
    #print(like);
    return like;
 
def lnpost(ws,fdensity,alpha,sig): 
    #converting flattened ws to matrix
    ws = ws.reshape((n_grid,n_grid));
    post = lnlike(ws) + lnprior(ws,fdensity,alpha,sig);
    return post;
    
def lnpost_k(ws,fdensity,alpha,sig): 
    #converting flattened ws to matrix
    #ws = real_to_complex(ws);
    ws = ws.reshape((n_grid,n_grid));
    ws = np.real(fft.ifft2(ws));
    post = lnlike_k(ws) + lnprior_k(ws,fdensity,alpha,sig);
    #print('post is');
    #print(post);
    return post;
  


def hess_k(ws,fdensity,alpha,sig,psf_k):
    print('hess_k begin');
    mo = np.exp(-4.);
    #ws = real_to_complex(ws);
    ws = ws.reshape((n_grid,n_grid));
    ws = np.absolute(fft.ifft2(ws));
    #calc l1
    l1 = -1*(psf_k**2 /sig_noise**2 / n_grid**2).flatten();
    #calc l2
    xsi = (1.-fdensity ) * gaussian(np.log(ws),loc=np.log(mo), scale=sig)/ws + fdensity*(ws**alpha /w_norm)
    dxsi = -1*gaussian(np.log(ws),loc=np.log(mo), scale=sig)*(1.-fdensity)/ws**2 - (1.-fdensity)*np.log(ws/mo)*np.exp(-np.log(ws/mo)**2 /2/sig**2)/np.sqrt(2*np.pi)/ws**2 /sig**3 + fdensity*alpha*ws**(alpha-1) /w_norm;
    dxsi_st = -1*gaussian(np.log(ws),loc=np.log(mo), scale=sig)*(1.-fdensity)/ws**2 - (1.-fdensity)*np.log(ws/mo)*np.exp(-np.log(ws/mo)**2 /2/sig**2)/np.sqrt(2*np.pi)/ws**2 /sig**3;
    ddxsi_st = -1*dxsi_st/ws - dxsi_st*np.log(ws/mo)/ws /sig**2 -(1.-fdensity)*(1/np.sqrt(2*np.pi)/sig)*np.exp(-np.log(ws/mo)**2 /2/sig**2)*(1/sig**2 - np.log(ws/mo)/sig**2 -1)/ ws**3;
    ddxsi = ddxsi_st + fdensity*alpha*(alpha-1)*ws**(alpha-2) /w_norm;
    l2 = -1*(dxsi/xsi)**2 + ddxsi/np.absolute(xsi);
    l2_k = fft.ifft2(l2).flatten()/n_grid**2;
    #we assume that hessian of l2 is diagonal. Under assumption k = -k', then we only get the zeroth element along the diag
    #lets fill the entire matrix and see whats up;
    hess_m = np.zeros((n_grid**2,n_grid**2),dtype=complex);
    hess_l1 = np.zeros((n_grid**2,n_grid**2),dtype=complex);
    np.fill_diagonal(hess_l1,l1);
    off = [];
    print(l2_k[0]);
    for i in range(0,n_grid**2):
        for j in range(0,n_grid**2):
            hess_m[i,j] = l2_k[int(np.absolute(i-j))];
            if i != j:
                off.append(l2_k[int(np.absolute(i-j))]);
    hess_m = hess_m + hess_l1;
    '''
    print('Sigma Real is:');
    print(np.std(np.real(off)));
    print('Simga Imag is:');
    print(np.std(np.imag(off)));
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(np.real(hess_m));
    ax[0].set_title('Real Hessian')
    #ax[1].imshow(data3[:-4,:-4]);
    ax[1].imshow(np.imag(hess_m));
    ax[1].set_title('Imaginary Hessian')
    plt.show();
    '''
    l_tot = np.diagonal(hess_m);
    l_minr = min(np.real(l_tot));
    l_mini = min(np.imag(l_tot))
    #print(l_tot-l1);
    if l_minr < 0:
        l_tot = l_tot - l_minr + 0.1;
    if l_mini < 0:
        l_tot = l_tot - 1j*(l_mini+0.1);
    '''
    print('diag is:');
    print(l2_k[0]);
    print('other is:');
    print(l1);
    '''
    '''
    hess_m = np.zeros((n_grid**2,n_grid**2));
    np.fill_diagonal(hess_m,l_tot);
    return hess_m;
    '''
    #return l1,l2_k[0];
    #l_tot =  complex_to_real(l_tot);
    #print('hess is');
    #print(l_tot);
    return l_tot;
def grad_k(ws,fdensity,alpha,sig,psf_k):
    print('grad_k begin');
    mo = np.exp(-4.);
    #ws = real_to_complex(ws);
    ws = ws.reshape((n_grid,n_grid));
    #wk = ws;
    ws = np.absolute(fft.ifft2(ws));
    
    #l1 = fft.ifft2((convolvesame_fft(ws,psf) - data)/sig_noise**2)#*psf_k;
    #l1_other = fft.ifft2((fft_convolve2d(ws,psf) - data)/sig_noise**2)#*fft.fft2(np.flipud(np.fliplr(psf)));
    l1 = -1*fft.ifft2((fft.ifft2(fft.fft2(ws)*fft.fft2(psf)) - data)/sig_noise**2)*psf_k;
    '''    
    l1_og = fft.ifft2((Psi(ws) - data)/sig_noise**2)*psf_k;
    l1_other = fft.ifft2((fft.fft2(fft.ifft2(ws)*fft.ifft2(psf)) - data)/sig_noise**2)*psf_k;
    print('diff is:');
    print(l1 - l1_other);
    print(l1-l1_og);
    '''
    #print(l1-l1_other)
    l1 = l1.flatten();
    
    xsi = (1.-fdensity ) * gaussian(np.log(ws),loc=np.log(mo), scale=sig)/ws + fdensity*(ws**alpha /w_norm)
    l2 = -1*gaussian(np.log(ws),loc=np.log(mo), scale=sig)*(1.-fdensity)/ws**2 - (1.-fdensity)*np.log(ws/mo)*np.exp(-np.log(ws/mo)**2 /2/sig**2)/np.sqrt(2*np.pi)/ws**2 /sig**3 + fdensity*alpha*ws**(alpha-1) /w_norm;
    l2 = l2/np.absolute(xsi);
    l2 = fft.ifft2(l2).flatten();
    l_tot = l1 + l2;
    #return l1,l2;
    #l_tot =  complex_to_real(l_tot);
    #print('grad is');
    #print(l_tot);
    return l_tot;

def optimize_m(t_ini, f_ini,alpha_ini, sig_curr,psf_k):
    #keeping in mind that minimize requires flattened arrays
    #afunc = Agrad.grad(lambda tt: -1*lnpost(tt, f_ini,alpha_ini, sig_curr));
    #grad_fun = lambda tg: grad_lnpost(tg,f_ini,alpha_ini,sig_curr);
    #hfunc = Agrad.hessian(lambda tt: -1*lnpost(tt, f_ini,alpha_ini, sig_curr));
    #hess_fun = lambda th: hess_lnpost(th,f_ini,alpha_ini,sig_curr);
    #grad_fun = Agrad.grad(lambda tg: -1*lnpost(tg,f_ini,alpha_ini,sig_curr));
    afunc = Agrad.holomorphic_grad(lambda tt: -1*lnpost_k(tt,f_curr,a_curr,sig_delta))
    grad_fun = lambda tg: -1*grad_k(tg,f_ini,alpha_ini,sig_curr,psf_k);
    hess_fun = lambda th: -1*hess_k(th,f_ini,alpha_ini,sig_curr,psf_k);
    
    '''
    res = scipy.optimize.minimize(lambda tt: -1*lnpost_k(tt,f_ini,alpha_ini,sig_curr),
                                  t_ini, # theta initial
                                  jac=afunc,
                                  method='BFGS')
    '''                              
        
    res = scipy.optimize.newton(lambda tt: np.zeros(n_grid**2)-lnpost_k(tt,f_ini,alpha_ini,sig_curr),t_ini,fprime=afunc,tol=1e-3,maxiter=150);
    tt_prime = np.real(fft.ifft(res));
    #print(tt_prime);
    '''                              
    tt_prime = res['x'];
    print('Number of Iterations:')
    print(res['nit'])
    print('Final Log-Likelihood:')
    '''
    w_final = tt_prime.reshape((n_grid,n_grid));
    #print(-1*lnpost(w_final,f_ini,alpha_ini,sig_curr));
    plt.imshow(w_final);
    plt.show();
    plt.imshow(data);
    plt.show();
    '''
    #print(w_final);
    #pick out the peaks using photutils
    thresh = detect_threshold(w_final,3);
    tbl = find_peaks(w_final,thresh);
    positions = np.transpose((tbl['x_peak'], tbl['y_peak']))
    w_peaks = np.zeros((n_grid,n_grid));
    w_peaks[positions] = w_final[positions];  
    '''
    return w_final;
'''   
def optimize_fa(t_ini, f_ini,alpha_ini, sig_curr):
    #keeping in mind that minimize requires flattened arrays
    res = scipy.optimize.minimize(Agrad.value_and_grad(lambda fa: -1.*lnpost(t_ini,fa[0],fa[1],sig_curr)),
                                  [f_ini,alpha_ini], # theta initial
                                  jac=True, 
                                  method='L-BFGS-B', 
                                  bounds=[(1e-5, 10)]*2)
                                  
    ff_prime,aa_prime = res['x'];
    return (ff_prime,aa_prime)
'''
########################################################################
#create mock data to run on
########################################################################
#create coordinate grid
theta_grid = np.linspace(0., 1., n_grid) # gridding of theta (same as pixels)

#create true values - assign to grid
x_true = np.abs(np.random.rand(Ndata)) # location of sources
y_true = np.abs(np.random.rand(Ndata));

#w_true = np.abs(np.random.rand(Ndata))+1;

#true grid needs to be set up with noise
w_true_grid = np.zeros((n_grid,n_grid))
for x,y, w in zip(x_true,y_true, w_true): 
    w_true_grid[np.argmin(np.abs(theta_grid - x)),np.argmin(np.abs(theta_grid - y))] = w

data4 = convolvesame_fft(w_true_grid,psf) + sig_noise# * np.random.randn(n_grid,n_grid);
data2 = Psi(w_true_grid) + sig_noise# * np.random.randn(n_grid,n_grid);
data3 = np.real(fft.ifft2(fft.fft2(np.pad(w_true_grid,((5,0),(5,0)),'constant'))*fft.fft2(np.pad(psf,((5,0),(5,0)),'constant')))) + sig_noise# * np.random.randn(n_grid,n_grid);
data = np.real(fft.ifft2(fft.fft2(w_true_grid)*fft.fft2(psf))) + sig_noise* np.random.randn(n_grid,n_grid);
#print(data-data2);
'''
fig, ax = plt.subplots(1,2)
ax[0].imshow(w_true_grid);
ax[0].set_title('True Positions')
#ax[1].imshow(data3[:-4,:-4]);
ax[1].imshow(data4);
ax[1].set_title('Observed Data')
plt.show();
'''
#create fft of psf
psf_k = fft.fft2(psf);
#psf_k = psf_k.flatten();
#print('psf_k is:');
#print(psf_k)
########################################################################
#now begin the actual execution
########################################################################


#now we begin the optimization
tt0 = np.zeros((n_grid,n_grid)) +3; #begin with high uniform M
#tt0 = np.absolute(np.random.randn(n_grid,n_grid)) + 2;
tt0 = fft.fft2(tt0).flatten();
#tt0 = complex_to_real(tt0);
#print(tt0);
#begin with the simple method of just minimizing
f_curr = fdensity_true;
a_curr = 2;
sig_delta = 0.75;
optimize_m(tt0,f_curr,a_curr,sig_delta,psf_k);
'''
#test of the smoothness of off-diags in hessian
for i in range(0,5):
    tt0 = np.absolute(np.random.randn(n_grid,n_grid))* 2;
    print(tt0);
    tt0 = fft.fft2(tt0).flatten();
    hess = hess_k(tt0,f_curr,a_curr,sig_delta,psf_k);
    off = np.where(~np.eye(hess.shape[0],dtype=bool));
    print('std is:')
    print(np.std(np.real(off)));
    print(np.std(np.imag(off)));
'''
'''
apfunc = Agrad.holomorphic_grad(lambda tt: -1*lnprior_k(tt,f_curr,a_curr,sig_delta));
alfunc = Agrad.holomorphic_grad(lambda tt: -1*lnlike_k(tt));
apval = apfunc(tt0);
alval = alfunc(tt0);
gval = -1*grad_k(tt0,f_curr,a_curr,sig_delta,psf_k);
#print(alval);
#print(glval);
#print(alval-glval);
#print(apval+gpval);
print(alval+apval - gval);
#print(gval);
'''

'''
bpfunc = Agrad.hessian(lambda tt: -1*lnprior_k(tt,f_curr,a_curr,sig_delta));
blfunc = Agrad.hessian(lambda tt: -1*lnlike_k(tt));
bpval = bpfunc(tt0);
blval = blfunc(tt0);

hlval,hpval = hess_k(tt0,f_curr,a_curr,sig_delta,psf_k)

print(bpval[0][:][:]);
print(blval[0][:][:]);
print(hpval);
print(hlval);
'''
#print(alval+glval);
#print(alval);
#print(glval);
#print(apval+gpval);

#a_prime, f_prime = optimize_fa(tt_prime,f_curr,a_curr,sig_delta);

