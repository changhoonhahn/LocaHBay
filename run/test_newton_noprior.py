import os 
import autograd as Agrad
import autograd.numpy as np 
import scipy.optimize
import autograd.numpy.fft as fft
import autograd.scipy.signal as signal
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

    
########################################################################
#create mock data to run on
########################################################################
np.random.seed(42)
Ndata = 1
n_grid = 5
pix_1d = np.linspace(0., 1., n_grid) # pixel gridding
fdensity_true = float(Ndata)/float(n_grid**2) #number density of obj in 1d

sig_psf = 0.01 # psf width
sig_noise = 0.01 # noise level

#create coordinate grid
theta_grid = np.linspace(0., 1., n_grid) # gridding of theta (same as pixels)

mid     = int(n_grid/2);
x, y    = np.meshgrid(theta_grid,theta_grid);
psf     = np.exp(-((y-theta_grid[mid])**2 + (x - theta_grid[mid])**2)/2./sig_psf**2)
#create fft of psf
psf_k = fft.fft2(psf)

#create true values - assign to grid
#x_true = np.abs(np.random.rand(Ndata)) # location of sources
#y_true = np.abs(np.random.rand(Ndata))
x_true = [0.5]
y_true = [0.5]
w_true = np.ones(Ndata) * 5 
#w_true = np.abs(np.random.rand(Ndata))+1

#true grid needs to be set up with noise
w_true_grid = np.zeros((n_grid,n_grid))
for x,y, w in zip(x_true,y_true, w_true): 
    w_true_grid[np.argmin(np.abs(theta_grid - x)),np.argmin(np.abs(theta_grid - y))] = w

w_true_k = fft.fft2(w_true_grid)
data = np.real(fft.ifft2(w_true_k * psf_k)) #+ np.absolute(sig_noise* np.random.randn(n_grid,n_grid))

data_p = signal.convolve(w_true_grid, psf)
diff = int((data_p.shape[0] - n_grid)/2)
data_p = data_p[diff:n_grid+diff,diff:n_grid+diff]

fig, ax = plt.subplots(1,3)
ax[0].imshow(w_true_grid)
ax[0].set_title('True Positions')
ax[1].imshow(data)
ax[1].set_title('data')
ax[2].imshow(data_p)
ax[2].set_title("data'")
plt.savefig('test.png')

def real_to_complex(z):      # real vector of length 2n -> complex of length n
    return z[:len(z)//2] + 1j * z[len(z)//2:]

def complex_to_real(z):      # complex vector of length n -> real of length 2n
    return np.real(np.concatenate((np.real(z), np.imag(z))))
    
def lnlike_k(ws): 
    ''' log likelihood w/ periodic boundary conditions (need for solving w/
    respect to fourier coefficients)
    '''
    return 0.5 * np.sum((np.real(fft.ifft2(ws*psf_k)) - data)**2)/sig_noise**2

def lnpost_k(ws,fdensity,alpha,sig): 
    #converting flattened ws to matrix
    ws = real_to_complex(ws)
    ws = ws.reshape((n_grid,n_grid))
    post = lnlike_k(ws) #+ lnprior_k(ws,fdensity,alpha,sig)
    return post

def lnpost_k_debug(ws, fdensity, alpha, sig): 
    #converting flattened ws to matrix
    print(ws[:5]) 
    ws = real_to_complex(ws)
    ws = ws.reshape((n_grid,n_grid))
    post = lnlike_k(ws) #+ lnprior_k(ws,fdensity,alpha,sig)
    
    fig, ax = plt.subplots(2,2)
    ax[0][0].imshow(np.real(fft.ifft2(w_true_k)), vmin=0, vmax=10)
    ax[0][0].set_title('Truth')
    ax[0][1].imshow(np.real(fft.ifft2(ws)), vmin=0, vmax=10)
    ax[1][0].imshow(data, vmin=0, vmax=10)
    ax[1][0].set_title('Data')
    ax[1][1].imshow(np.real(fft.ifft2(ws*psf_k)), vmin=0, vmax=10)

    ii = 0 
    while os.path.isfile('test_opt%i.png' % ii): 
        ii += 1 
    plt.savefig('test_opt%i.png' % ii)
    plt.close()
    return post
    
def lnpost_k_og(ws, fdensity, alpha, sig): 
    #converting flattened ws to matrix
    ws = ws.reshape((n_grid,n_grid))
    post = lnlike_k(ws) #+ lnprior_k(ws,fdensity,alpha,sig)
    return post  

#function for determining the hessian w/ respect to fourier coeff
def hess_k(ws,fdensity,alpha,sig):
    #print('hess_k begin')
    #mo = np.exp(-4.)
    #ws = real_to_complex(ws)
    #ws = ws.reshape((n_grid,n_grid))
    #ws = np.real(fft.ifft2(ws))
    #calc l1 we only get diagonals here
    l1 = -1*(psf_k**2 /sig_noise**2 / n_grid**2).flatten()
    hess_l1 = np.zeros((2*n_grid**2,2*n_grid**2),dtype=complex)
    np.fill_diagonal(hess_l1,complex_to_real(l1))
    l_tot = hess_l1
    #print('hess is:')
    #print(l_tot)
    return l_tot

def grad_k(ws,fdensity,alpha,sig):
    #print('grad_k begin')
    mo = np.exp(-4.)
    ws = real_to_complex(ws)
    ws = ws.reshape((n_grid,n_grid))
    #wk = ws
    #ws = np.real(fft.ifft2(ws))
    
    l1 = -1*fft.ifft2((np.real(fft.ifft2(ws*psf_k))- data)/sig_noise**2)*psf_k
    #print(l1-l1_other)
    l1 = l1.flatten()
    l_tot = l1
    #return l1,l2
    l_tot =  complex_to_real(l_tot)
    return l_tot


def optimize_m(t_ini, f_ini, alpha_ini, sig_curr):
    #keeping in mind that minimize requires flattened arrays
    print('Initial Likelihood')
    print(lnpost_k(t_ini,f_ini,alpha_ini,sig_curr))
    t_ini_comp = real_to_complex(t_ini)

    hfunc = Agrad.hessian(lambda tt: lnpost_k(tt, f_ini,alpha_ini, sig_curr))
    afunc = Agrad.grad(lambda tt: lnpost_k(tt,f_curr,a_curr,sig_delta))
    grad_fun = lambda tg: -1*grad_k(tg,f_ini,alpha_ini,sig_curr)
    hess_fun = lambda th: -1*hess_k(th,f_ini,alpha_ini,sig_curr)
    afunc_og = Agrad.holomorphic_grad(lambda tt: np.conj(lnpost_k_og(tt,f_curr,a_curr,sig_delta)))
    aog = lambda ts: complex_to_real(afunc_og(real_to_complex(ts)))

    #try optimization with some different algorithms
    res = scipy.optimize.minimize(lambda tt: lnpost_k(tt,f_ini,alpha_ini,sig_curr),
                                  t_ini, # theta initial
                                  jac=grad_fun,
                                  hess = hess_fun,
                                  method='trust-ncg') 
    res2 = scipy.optimize.minimize(lambda tt: lnpost_k(tt,f_ini,alpha_ini,sig_curr),
                                  t_ini, # theta initial
                                  jac=aog,
                                  method='CG')            
    res3 = scipy.optimize.minimize(
            lambda tt: lnpost_k(tt, f_ini, alpha_ini, sig_curr),
            t_ini, 
            method='Nelder-Mead') 
                              
    #cres = real_to_complex(res['x'])
    #tt_prime = np.real(fft.ifft(cres))
    #cres2 = real_to_complex(res2.x)
    #tt_prime2 = np.real(fft.ifft(cres2))
    print('Final Log Likelihood')
    print(lnpost_k(res.x,f_ini,alpha_ini,sig_curr))
    print(lnpost_k(res2.x,f_ini,alpha_ini,sig_curr))
    
    w_final_k = real_to_complex(res['x']).reshape(n_grid, n_grid)
    w_final = np.real(fft.ifft2(w_final_k))
    w_final2_k = real_to_complex(res2.x).reshape(n_grid, n_grid)
    w_final2 = np.real(fft.ifft2(w_final2_k))
    w_final3_k = real_to_complex(res3.x).reshape(n_grid, n_grid)
    w_final3 = np.real(fft.ifft2(w_final3_k))

    fig, ax = plt.subplots(2,4)
    ax[0][0].imshow(np.real(fft.ifft2(w_true_k)), vmin=0, vmax=10)
    ax[0][0].set_title('Truth')
    ax[0][1].imshow(w_final2, vmin=0, vmax=10)
    ax[0][1].set_title('Newton-CG')
    ax[0][2].imshow(w_final, vmin=0, vmax=10)
    ax[0][2].set_title('trust-ncg')
    ax[0][3].imshow(w_final3, vmin=0, vmax=10)
    ax[0][3].set_title('nelder-mead')
    ax[1][0].imshow(data, vmin=0, vmax=10)
    #ax[1][0].imshow(np.real(fft.ifft2(w_true_k*psf_k)))
    ax[1][0].set_title('Data')
    ax[1][1].imshow(np.real(fft.ifft2(w_final2_k*psf_k)), vmin=0, vmax=10)
    ax[1][1].set_title('FM Newton-CG')
    ax[1][2].imshow(np.real(fft.ifft2(w_final_k*psf_k)), vmin=0, vmax=10)
    ax[1][2].set_title('FM trust-ncg')
    ax[1][3].imshow(np.real(fft.ifft2(w_final3_k*psf_k)), vmin=0, vmax=10)
    ax[1][3].set_title('FM nelder-mead')
    plt.savefig('test_opt.png')
    return w_final
########################################################################
#now begin the actual execution
########################################################################

#now we begin the optimization
tt0 = np.zeros((n_grid,n_grid)) +6 #begin with high uniform M
#tt0 = np.absolute(np.random.randn(n_grid,n_grid)) + 2 #for test case with non uniform initial conditions
#tt0 = w_true_grid
tt0 = fft.fft2(tt0).flatten()
tto = tt0
tt0 = complex_to_real(tt0)
#tt0 = np.where(np.abs(tt0)>1e-5, tt0, 0)
#print(tt0)
#begin with the simple method of just minimizing
f_curr = fdensity_true
a_curr = 2
sig_delta = 0.75
#print(lnpost_k(tt0,f_curr,a_curr,sig_delta))

print(optimize_m(tt0, f_curr, a_curr, sig_delta)) 

#tt_true = complex_to_real(w_true_k.flatten()) 
#print(lnpost_k_debug(tt_true, f_curr, a_curr, sig_delta))
