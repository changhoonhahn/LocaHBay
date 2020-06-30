import math as math
import numpy as np 
#import numpy as np
import numpy.fft as fft
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
n_grid = 6;
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
    print('len of z:');
    print(len(z));
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
#zero pad true_grid and psf
'''
wtg = np.zeros((9,9));
wtg[2:7,2:7] = w_true_grid;
p = np.zeros((9,9));
p[2:7,2:7] = psf;
'''
data = np.real(fft.ifft2(fft.fft2(w_true_grid)*fft.fft2(psf))) + sig_noise* np.random.randn(n_grid,n_grid);
data2 = Psi(w_true_grid) + sig_noise * np.random.randn(n_grid,n_grid);
data3 = signal.convolve(w_true_grid,psf);
diff = int((len(data3[:,0]) - n_grid)/2);
data3 = data3[diff:n_grid+diff,diff:n_grid+diff]
#data4 = np.real(fft.ifft2(fft.fft2(wtg)*fft.fft2(p)))
data4 = np.real(fft.ifft2(fft.ifft2(w_true_grid)))*n_grid**2;
fig, ax = plt.subplots(1,2)
ax[0].imshow(w_true_grid);
ax[0].set_title('True Positions')
#ax[1].imshow(data3[:-4,:-4]);
ax[1].imshow(data4);
ax[1].set_title('Observed Data')
plt.show();

#create fft of psf
psf_k = fft.fft2(psf);


########################################################################
#now begin the actual execution
########################################################################


#now we begin the optimization
tt0 = np.zeros((n_grid,n_grid)) +3; #begin with high uniform M
#tt0 = np.absolute(np.random.randn(n_grid,n_grid)) + 2; #for test case with non uniform initial conditions

tt0 = fft.fft2(tt0).flatten();
tto = tt0;
tt0 = complex_to_real(tt0);