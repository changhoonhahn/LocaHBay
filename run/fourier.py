import scipy.optimize
import autograd as Agrad
import autograd.numpy as np 
import autograd.numpy.fft as fft
import autograd.scipy.signal as aSignal
from scipy import signal as Signal

import matplotlib.pyplot as plt 


np.random.seed(42)
Ndata   = 1
n_grid  = 10
grid1d  = np.linspace(0., 1., n_grid)
fdensity_true = float(Ndata)/float(n_grid**2); #number density of obj in 1d

sig_psf     = 0.01 # psf width
sig_noise   = 0.01 # noise level

#create true values - assign to grid
#x_true = np.abs(np.random.rand(Ndata)) # location of sources
#y_true = np.abs(np.random.rand(Ndata));
x_true = [0.5]
y_true = [0.5]
w_true = np.ones(Ndata) * 5
print(x_true, y_true, w_true)

#true grid needs to be set up with noise
w_true_grid = np.zeros((n_grid,n_grid))
for x, y, w in zip(x_true, y_true, w_true): 
    w_true_grid[(np.abs(grid1d - x)).argmin(), (np.abs(grid1d - y)).argmin()] = w

mid     = int(n_grid/2);
x, y    = np.meshgrid(grid1d,grid1d);
psf     = np.exp(-((y-grid1d[mid])**2 + (x - grid1d[mid])**2)/2./sig_psf**2)

xx = np.linspace(-0.5, 0.5, n_grid) 
bump =  np.exp(-0.5*xx**2/sig_psf**2) 
#bump /= np.trapz(bump) # normalize the integral to 1
_psf = bump[:, np.newaxis] * bump[np.newaxis,:]

fig, ax = plt.subplots(1,2)
ax[0].imshow(psf)
ax[0].set_title('psf')
ax[1].imshow(_psf)
ax[1].set_title('psf')
plt.savefig('test0.png')

data = np.real(fft.ifft2(fft.fft2(w_true_grid)*fft.fft2(_psf))) 
data_p = Signal.fftconvolve(w_true_grid, _psf)#, mode='same')
#psf_k = fft.fft2(psf);
psf_k = fft.fft2(_psf);

def lnlike_k(ws): 
    ''' log likelihood w/ periodic boundary conditions (need for solving w/
    respect to fourier coefficients)
    '''
    return 0.5 * np.sum((aSignal.convolve(ws, _psf) - data_p)**2)/sig_noise**2
    #return 0.5 * np.sum(((fft.ifft2(ws*psf_k))- data_p)**2)/sig_noise**2
    #return 0.5 * np.sum((np.real(fft.ifft2(ws*psf_k))- data)**2)/sig_noise**2

def lnpost_k(ws):
    ''' just the likelihood for now 
    '''
    print(ws) 
    ws = ws.reshape(n_grid, n_grid)
    return  lnlike_k(ws)

def optimize_m(t_ini, f_ini, alpha_ini, sig_curr, psf_k):
    #keeping in mind that minimize requires flattened arrays
    print('Initial Likelihood')
    print(lnpost_k(t_ini))
    
    afunc = Agrad.grad(lambda tt: lnpost_k(tt))

    res = scipy.optimize.minimize(
            lambda tt: lnpost_k(tt),
            t_ini, # theta initial
            jac=afunc,
            method='CG')            
    tt = res.x 
    print('Final Log Likelihood')
    print(lnpost_k(res.x))

    tt = res.x 
    w_final = tt.reshape((n_grid,n_grid));

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(data);
    ax[0].set_title('Truth')
    ax[1].imshow(w_final);
    ax[1].set_title('Newton-CG')
    plt.savefig('test_opt.png')
    return w_final;

fig, ax = plt.subplots(1,3)
ax[0].imshow(w_true_grid)
ax[0].set_title('True Positions')
ax[1].imshow(data)
ax[1].set_title('Observed Data')
ax[2].imshow(data_p)
ax[2].set_title('Observed Data')
plt.savefig('test.png')

#now we begin the optimization
#tt0 = np.zeros((n_grid,n_grid)) +6 #begin with high uniform M
#tto = fft.fft2(tt0).flatten()
#tt0 = complex_to_real(tt0)
tt0 = np.tile(6, (n_grid, n_grid)).flatten()
print(tt0) 

#begin with the simple method of just minimizing
f_curr = fdensity_true
a_curr = 2
sig_delta = 0.75

optimize_m(tt0, f_curr, a_curr, sig_delta, psf_k)
