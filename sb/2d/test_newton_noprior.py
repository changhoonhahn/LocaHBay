
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
Ndata = 3;
n_grid =10;
pix_1d = np.linspace(0., 1., n_grid) # pixel gridding
fdensity_true = float(Ndata)/float(n_grid**2); #number density of obj in 1d

sig_psf = 0.01 # psf width
sig_noise = 0.01 # noise level

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
    
def real_to_complex(z):      # real vector of length 2n -> complex of length n
    #print('len of z:');
    #print(len(z));
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
        p1 = w**alpha /w_norm; #probability of single source
        pri += fdensity#*p1 #+ p2*fdensity**2
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
    ''' log likelihood (non-periodic boundary)
    '''
    return -0.5 * np.sum((Psi(ws) - data)**2/sig_noise**2)
    
def lnprior_k(ws,fdensity,alpha,sig): 
    #ws = np.absolute(fft.ifft(ws));
    #ws = ws.reshape((n_grid,n_grid));
    #print('priori');
    #print(ws);
    pri =  np.sum([np.log(prior_i(w,fdensity,alpha,sig)) for w in ws.flatten()]);
    return pri;

def lnlike_k(ws): 
    ''' log likelihood w/ periodic boundary conditions (need for solving w/
    respect to fourier coefficients)
    '''
    #cv = signal.convolve(ws,psf);
    #diff = int((len(cv[:,0]) - n_grid)/2);
    #cv = cv[diff:n_grid+diff,diff:n_grid+diff];
    #like = -0.5 * np.sum((cv - data)**2/sig_noise**2);
    #like = -0.5 * np.sum((np.real(fft.fft2(fft.fft2(ws)))- data)**2/sig_noise**2);
    like = 0.5 * np.sum((np.real(fft.ifft2(ws*psf_k))- data)**2)/sig_noise**2;
    #like = np.real(like);
    #print('like is:');
    #print(like);
    return like;
'''    
def lnlike_k_c(ws): 

    #ws = ws.reshape((n_grid,n_grid));
    #ws = np.absolute(fft.ifft2(ws));
    ws = real_to_complex(ws);
    ws = ws.reshape((n_grid,n_grid));
    ws = np.real(fft.ifft2(ws));
    #like = -0.5 * np.sum((np.real(fft.ifft2(fft.fft2(ws)*fft.fft2(psf))) - data)**2/sig_noise**2);
    like = -0.5 * np.sum((np.real(fft.fft2(fft.fft2(ws)))- data)**2/sig_noise**2);
    #like = np.real(like);
    #print('like is:');
    #print(like);
    return like;
''' 
def lnpost(ws,fdensity,alpha,sig): 
    #converting flattened ws to matrix
    ws = ws.reshape((n_grid,n_grid));
    post = lnlike(ws) + lnprior(ws,fdensity,alpha,sig);
    return post;
    
def lnpost_k(ws,fdensity,alpha,sig): 
    #converting flattened ws to matrix
    ws = real_to_complex(ws);
    ws = ws.reshape((n_grid,n_grid));
    post = lnlike_k(ws) #+ lnprior_k(ws,fdensity,alpha,sig);
    #print('post is');
    #print(post);
    return post;
    
def lnpost_k_og(ws,fdensity,alpha,sig): 
    #converting flattened ws to matrix
    #print(np.shape(ws))
    ws = ws.reshape((n_grid,n_grid));
    #ws = np.real(fft.ifft2(ws));
    post = lnlike_k(ws) #+ lnprior_k(ws,fdensity,alpha,sig);
    #print('post is');
    #print(post);
    return post;  

#function for determining the hessian w/ respect to fourier coeff
def hess_k(ws,fdensity,alpha,sig,psf_k):
    #print('hess_k begin');
    #mo = np.exp(-4.);
    #ws = real_to_complex(ws);
    #ws = ws.reshape((n_grid,n_grid));
    #ws = np.real(fft.ifft2(ws));
    #calc l1 we only get diagonals here
    l1 = -1*(psf_k**2 /sig_noise**2 / n_grid**2).flatten();
    hess_l1 = np.zeros((2*n_grid**2,2*n_grid**2),dtype=complex);
    np.fill_diagonal(hess_l1,complex_to_real(l1));
    l_tot = hess_l1;
    #print('hess is:');
    print(l_tot);
    return l_tot;
def grad_k(ws,fdensity,alpha,sig,psf_k):
    #print('grad_k begin');
    mo = np.exp(-4.);
    ws = real_to_complex(ws);
    ws = ws.reshape((n_grid,n_grid));
    #wk = ws;
    #ws = np.real(fft.ifft2(ws));
    
    l1 = -1*fft.ifft2((np.real(fft.ifft2(ws*psf_k))- data)/sig_noise**2)*psf_k;
    #print(l1-l1_other)
    l1 = l1.flatten();
    l_tot = l1;
    #return l1,l2;
    l_tot =  complex_to_real(l_tot);
    #print('grad is');
    #print(l_tot);
    return l_tot;


def optimize_m(t_ini, f_ini,alpha_ini, sig_curr,psf_k):
    #keeping in mind that minimize requires flattened arrays
    print('Initial Likelihood');
    print(lnpost_k(t_ini,f_ini,alpha_ini,sig_curr))
    t_ini_comp = real_to_complex(t_ini)
    hfunc = Agrad.hessian(lambda tt: lnpost_k(tt, f_ini,alpha_ini, sig_curr));
    afunc = Agrad.grad(lambda tt: lnpost_k(tt,f_curr,a_curr,sig_delta))
    grad_fun = lambda tg: -1*grad_k(tg,f_ini,alpha_ini,sig_curr,psf_k);
    hess_fun = lambda th: -1*hess_k(th,f_ini,alpha_ini,sig_curr,psf_k);
    afunc_og = Agrad.holomorphic_grad(lambda tt: np.conj(lnpost_k_og(tt,f_curr,a_curr,sig_delta)));
    aog = lambda ts: complex_to_real(afunc_og(real_to_complex(ts)));
    #try optimization with some different algorithms
    res = scipy.optimize.minimize(lambda tt: lnpost_k(tt,f_ini,alpha_ini,sig_curr),
                                  t_ini, # theta initial
                                  jac=grad_fun,
                                  hess = hess_fun,
                                  method='trust-ncg') 
    '''                            
    res2 = scipy.optimize.minimize(lambda tt: -1*lnpost_k(tt,f_ini,alpha_ini,sig_curr),
                                  t_ini, # theta initial
                                  jac=afunc,
                                  hess = hfunc,
                                  method='Newton-CG') 
                                  
    '''
    '''                              
    res2 = scipy.optimize.minimize(lambda tt: -1*lnpost_k(tt,f_ini,alpha_ini,sig_curr),
                                  t_ini, # theta initial
                                  jac=grad_fun,
                                  method='CG')   
    '''                              
    '''
    res2 = scipy.optimize.minimize(lambda tt: -1*lnpost_k(tt,f_ini,alpha_ini,sig_curr),
                              t_ini, # theta initial
                              method='Nelder-Mead')                                  
                          
    '''
    
    res2 = scipy.optimize.minimize(lambda tt: lnpost_k(tt,f_ini,alpha_ini,sig_curr),
                                  t_ini, # theta initial
                                  jac=aog,
                                  method='CG')            
                              
      
    
    
    cres = real_to_complex(res['x'])
    tt_prime = np.real(fft.ifft(cres));
    cres2 = real_to_complex(res2.x)
    tt_prime2 = np.real(fft.ifft(cres2));
    print('Final Log Likelihood')
    print(lnpost_k(res.x,f_ini,alpha_ini,sig_curr))
    print(lnpost_k(res2.x,f_ini,alpha_ini,sig_curr))
    '''                              
    tt_prime = res['x'];
    print('Number of Iterations:')
    print(res['nit'])
    print('Final Log-Likelihood:')
    '''
    w_final = tt_prime.reshape((n_grid,n_grid));
    w_final2 = tt_prime2.reshape((n_grid,n_grid));
    tt_sum = np.sum(w_final,axis=0);
    print(tt_sum);
    print(np.sum(w_true_grid,axis=1));
    #plt.imshow(tt_sum);
    #print(-1*lnpost(w_final,f_ini,alpha_ini,sig_curr));
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(data);
    ax[0].set_title('Truth')
    ax[1].imshow(w_final2);
    ax[1].set_title('Newton-CG')
    ax[2].imshow(w_final);
    ax[2].set_title('trust-ncg')
    plt.show();

    return w_final;
    
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

data = np.real(fft.ifft2(fft.fft2(w_true_grid)*fft.fft2(psf))) #+ np.absolute(sig_noise* np.random.randn(n_grid,n_grid));
data3 = signal.convolve(w_true_grid,psf);
diff = int((len(data3[:,0]) - n_grid)/2);
data3 = data3[diff:n_grid+diff,diff:n_grid+diff];
#data = data3;
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


########################################################################
#now begin the actual execution
########################################################################


#now we begin the optimization
tt0 = np.zeros((n_grid,n_grid)) +6; #begin with high uniform M
#tt0 = np.absolute(np.random.randn(n_grid,n_grid)) + 2; #for test case with non uniform initial conditions
#tt0 = w_true_grid;
tt0 = fft.fft2(tt0).flatten();
tto = tt0;
tt0 = complex_to_real(tt0);
#tt0 = np.where(np.abs(tt0)>1e-5, tt0, 0)
#print(tt0);
#begin with the simple method of just minimizing
f_curr = fdensity_true;
a_curr = 2;
sig_delta = 0.75;
#print(lnpost_k(tt0,f_curr,a_curr,sig_delta));

optimize_m(tt0,f_curr,a_curr,sig_delta,psf_k);

#print(lnpost_k_og(real_to_complex(tt0),f_curr,a_curr,sig_delta))
'''
#below are a bunch of tests I did for checking my analytics vs autograd
#test gradient with autograd (and maybe hessian too) in our new realtocomplex format
afunc = Agrad.grad(lambda tt: -1*lnpost_k(tt,f_curr,a_curr,sig_delta));
afunc_og = Agrad.holomorphic_grad(lambda tt: -1*lnpost_k_og(tt,f_curr,a_curr,sig_delta));
gval = -1*grad_k(tt0,f_curr,a_curr,sig_delta,psf_k);
aval = afunc(tt0);
aval_og = afunc_og(tto);
#print(real_to_complex(aval) - real_to_complex(gval));
print(aval_og)
print( real_to_complex(aval));
'''
'''
#test accuracy of likelihood grad

afunc = Agrad.grad(lambda tt: -1*lnlike_k_c(tt));
afunc_og = Agrad.holomorphic_grad(lambda tt: -1*lnlike_k(tt));

print('like is');
print(lnlike_k_c(tt0));
print(lnlike_k(tto))
aval = afunc(tt0);
aval_og = afunc_og(tto);
print(real_to_complex(aval));
print(aval_og)
'''
'''
#test accuracy of hessian

afunc = Agrad.hessian(lambda tt: -1*lnpost_k(tt,f_curr,a_curr,sig_delta));
aval = afunc(tt0);
hval = real_to_complex(-1*hess_k(tt0,f_curr,a_curr,sig_delta,psf_k));
print(np.shape(aval));
print(np.shape(hval));

adiag = real_to_complex(np.diagonal(aval[0][:][:]));
print(adiag);
print(hval);
'''


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
#print(alval+glval);
#print(alval);
#print(glval);
#print(apval+gpval);

#a_prime, f_prime = optimize_fa(tt_prime,f_curr,a_curr,sig_delta);

