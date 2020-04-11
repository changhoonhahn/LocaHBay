import math as math
import numpy as np
import numpy.fft as fft
import scipy.optimize
import scipy.stats as st


np.random.seed(42)
sig_psf = np.linspace(0.01,0.5,10);
sig_noise = np.linspace(0.01,0.5,10);

Ndata = 5;
n_grid = 15;
pix_1d = np.linspace(0., 1., n_grid) # pixel gridding
fdensity_true = float(Ndata)/float(n_grid**2); #number density of obj in 1d

#create coordinate grid
theta_grid = np.linspace(0., 1., n_grid) # gridding of theta (same as pixels)

#create true values - assign to grid
x_true = np.abs(np.random.rand(Ndata)) # location of sources
y_true = np.abs(np.random.rand(Ndata));

#these are values for the power law function for sampling intensities
w_interval = (1,2);
w_lin = np.linspace(1,2,100);
alpha_true = 2;
w_norm = (50**(alpha_true+1) - w_interval[0]**(alpha_true+1))/(alpha_true+1);
w_func = np.power(w_lin,alpha_true)/w_norm;
w_true = w_norm*np.random.choice(w_func,Ndata);
w_true_grid = np.zeros((n_grid,n_grid))

for x,y, w in zip(x_true,y_true, w_true): 
    w_true_grid[np.argmin(np.abs(theta_grid - x)),np.argmin(np.abs(theta_grid - y))] = w;

def psi(pos,sig_p): 
    ''' measurement model, which in our case is just a 1d gaussian of width 
    sigma (PSF) written out to a meshgrid created by pix1d 
    '''
    x,y = np.meshgrid(pix_1d,pix_1d);
    return np.exp(-((y-pix_1d[pos[0]])**2 + (x - pix_1d[pos[1]])**2)/2/sig_p**2); #keep in mind difference between x and y position and indices! Here, you are given indices, but meshgrid is in x-y coords

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
    
    
def Psi(ws,sig_p): 
    ''' "forward operator" i.e. forward model 
    
    Psi = int psi(theta) dmu(theta) 

    where mu is the signal parameter
    '''
    return np.sum(np.array([w*psi(index,sig_p) for (index,w) in np.ndenumerate(ws)]),0)





#true grid needs to be set up with noise
data = np.zeros((n_grid,n_grid, len(sig_psf)*len(sig_noise)));
for i in range(0,len(sig_psf)):
    for j in range(0,len(sig_noise)):
        data[:,:,i*len(sig_noise) + j] = Psi(w_true_grid,sig_psf[i]) + sig_noise[j] * np.random.randn(n_grid,n_grid);
        fname = './data_5_15/test' + str(i*len(sig_noise) + j)
        np.savetxt(fname+'.dat',data[:,:,i*len(sig_noise) + j]);
np.savetxt('true.dat',w_true_grid);