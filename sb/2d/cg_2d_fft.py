
import math as math
import autograd as Agrad
import autograd.numpy as np 
import autograd.numpy.fft as fft
#import numpy as np
import scipy.optimize
import scipy.stats as st
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
cg_2d.py
Code modified from rp_2d.py
Author: Massimo Pascale
Last Updated: 3/1/2020

Code uses poisson prior and exponential intensity function to determine
point source locations in psf+noise and recover hyperparameters.
'''
########################################################################




#create global definitions - this will become a main function later on
np.random.seed(42)
Ndata = 1;
n_grid = 15;
pix_1d = np.linspace(0., 1., n_grid) # pixel gridding
fdensity_true = float(Ndata)/float(n_grid**2); #number density of obj in 1d

sig_psf = 0.01 # psf width
sig_noise = 0.01 # noise level

mid = int(n_grid/2);
x,y = np.meshgrid(pix_1d,pix_1d);
psf = np.exp(-((y-pix_1d[mid])**2 + (x - pix_1d[mid])**2)/2/sig_psf**2); #keep in mind difference between x and y position and indices! Here, you are given indices, but meshgrid is in x-y coords

#these are values for the power law function for sampling intensities
w_interval = (1,2);
w_lin = np.linspace(1,2,100);
alpha_true = 2;
w_norm = (50**(alpha_true+1) - w_interval[0]**(alpha_true+1))/(alpha_true+1);
w_func = np.power(w_lin,alpha_true)/w_norm;
w_true = w_norm*np.random.choice(w_func,Ndata);


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
    return -0.5 * np.sum((np.real(fft.ifft2(fft.fft2(ws)*fft.fft2(psf))) - data)**2/sig_noise**2)
 
def lnpost(ws,fdensity,alpha,sig): 
    #converting flattened ws to matrix
    ws = ws.reshape((n_grid,n_grid));
    post = lnlike(ws) + lnprior(ws,fdensity,alpha,sig);
    return post;

def grad_lnpost(ws,fdensity,alpha,sig):
    #calculate gradient of the ln posterior
    print('grad');
    mo = np.exp(-4.);
    ws = ws.reshape((n_grid,n_grid));
    #calc l1
    bsis = (Psi(ws)-data)/sig_noise**2;
    lsis = ws*0;
    for (index,w) in np.ndenumerate(ws):
        lsis[index] = np.sum(psi(index)*bsis);
    l1 = lsis#*np.sum((Psi(ws)-data)/2/sig_noise**2);
    xsi = (1.-fdensity ) * gaussian(np.log(ws),loc=np.log(mo), scale=sig)/ws + fdensity*(ws**alpha /w_norm)
    l2 = -1*gaussian(np.log(ws),loc=np.log(mo), scale=sig)*(1.-fdensity)/ws**2 - (1.-fdensity)*np.log(ws/mo)*np.exp(-np.log(ws/mo)**2 /2/sig**2)/np.sqrt(2*np.pi)/ws**2 /sig**3 + fdensity*alpha*ws**(alpha-1) /w_norm;
    l2 = l2/np.absolute(xsi);
    l_tot = l1-l2;
    return l_tot.flatten();
    #aval = afunc(ws);
    #print('new it')
    #print(np.absolute(aval-l_tot));
    
def hess_lnpost(ws,fdensity,alpha,sig):
    print('hess')
    #print(ws);
    mo = np.exp(-4.);
    #hval = hfunc(ws);
    ws = ws.reshape((n_grid,n_grid));
    #calc l1
    lsis = np.array([-1*np.sum(psi(index)**2)/sig_noise**2 for (index,w) in np.ndenumerate(ws)]);
    lsis = lsis.reshape((n_grid,n_grid));
    l1 = lsis#*np.sum((Psi(ws)-data)/2/sig_noise**2);
    xsi = (1.-fdensity ) * gaussian(np.log(ws),loc=np.log(mo), scale=sig)/ws + fdensity*(ws**alpha /w_norm)
    dxsi = -1*gaussian(np.log(ws),loc=np.log(mo), scale=sig)*(1.-fdensity)/ws**2 - (1.-fdensity)*np.log(ws/mo)*np.exp(-np.log(ws/mo)**2 /2/sig**2)/np.sqrt(2*np.pi)/ws**2 /sig**3 + fdensity*alpha*ws**(alpha-1) /w_norm;
    dxsi_st = -1*gaussian(np.log(ws),loc=np.log(mo), scale=sig)*(1.-fdensity)/ws**2 - (1.-fdensity)*np.log(ws/mo)*np.exp(-np.log(ws/mo)**2 /2/sig**2)/np.sqrt(2*np.pi)/ws**2 /sig**3;
    ddxsi_st = -1*dxsi_st/ws - dxsi_st*np.log(ws/mo)/ws /sig**2 -(1.-fdensity)*(1/np.sqrt(2*np.pi)/sig)*np.exp(-np.log(ws/mo)**2 /2/sig**2)*(1/sig**2 - np.log(ws/mo)/sig**2 -1)/ ws**3;
    ddxsi = ddxsi_st + fdensity*alpha*(alpha-1)*ws**(alpha-2) /w_norm   ;
    l2 = -1*(dxsi/xsi)**2 + ddxsi/np.absolute(xsi);
    l_tot = l1+l2;
    #those are the diagonal terms, now need to build off diagonal
    hess_m = np.zeros((n_grid**2,n_grid**2));
    np.fill_diagonal(hess_m,l_tot);
    '''
    for i in range(0,n_grid**2):
        for j in range(i+1,n_grid**2):
            ind1 = (int(i/n_grid),i%n_grid);
            ind2 = (int(j/n_grid),j%n_grid);
            hess_m[i,j] = -1*np.sum(psi(ind1)*psi(ind2))/sig_noise**2;

    hess_m = symmetrize(hess_m);
    '''
    print('hess fin');
    #print(l_tot);
    #print('new it');
    #print(np.average(hval[0][:][:]-hess_m));
    return -1*hess_m;
def optimize_m(t_ini, f_ini,alpha_ini, sig_curr):
    #keeping in mind that minimize requires flattened arrays
    afunc = Agrad.grad(lambda tt: -1*lnpost(tt, f_ini,alpha_ini, sig_curr));
    grad_fun = lambda tg: grad_lnpost(tg,f_ini,alpha_ini,sig_curr);
    #hfunc = Agrad.hessian(lambda tt: -1*lnpost(tt, f_ini,alpha_ini, sig_curr));
    hess_fun = lambda th: hess_lnpost(th,f_ini,alpha_ini,sig_curr);
    #grad_fun = Agrad.grad(lambda tg: -1*lnpost(tg,f_ini,alpha_ini,sig_curr));
    '''
    res = scipy.optimize.minimize(lambda tt: -1*lnpost(tt,f_ini,alpha_ini,sig_curr),
                                  t_ini, # theta initial
                                  jac=afunc, 
                                  method='L-BFGS-B', 
                                  bounds=[(1e-5, 10)]*len(t_ini))
    '''                              
                                  
    '''                                   
    res = scipy.optimize.minimize(lambda tt: -1*lnpost(tt,f_ini,alpha_ini,sig_curr),
                                  t_ini, # theta initial
                                  jac=grad_fun,
                                  hess = hess_fun,
                                  method='trust-ncg');         
    '''
    res = scipy.optimize.minimize(lambda tt: -1*lnpost(tt,f_ini,alpha_ini,sig_curr),
                                  t_ini, # theta initial
                                  method='Nelder-Mead')                              
    tt_prime = res['x'];
    print('Number of Iterations:')
    print(res['nit'])
    print('Final Log-Likelihood:')
    w_final = tt_prime.reshape((n_grid,n_grid));
    print(-1*lnpost(w_final,f_ini,alpha_ini,sig_curr));
    plt.imshow(w_final);
    plt.show();
    plt.imshow(w_true_grid);
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
data =np.real(fft.ifft2(fft.fft2(w_true_grid)*fft.fft2(psf)))+ sig_noise * np.random.randn(n_grid,n_grid);

########################################################################
#now begin the actual execution
########################################################################


#now we begin the optimization
tt0 = np.zeros(n_grid**2) +3; #begin with high uniform M

#begin with the simple method of just minimizing
f_curr = fdensity_true;
a_curr = 2;
sig_delta = 0.75;
posterior = lnpost(tt0,f_curr,a_curr,sig_delta);
print(posterior)
tt_prime = optimize_m(tt0,f_curr,a_curr,sig_delta);

#a_prime, f_prime = optimize_fa(tt_prime,f_curr,a_curr,sig_delta);

fig, ax = plt.subplots(1,3)
ax[0].imshow(w_true_grid);
ax[0].set_title('True Positions')
ax[1].imshow(data);
ax[1].set_title('Observed Data')
ax[2].imshow(tt_prime);
ax[2].set_title('Sparse Bayes')
plt.show();
