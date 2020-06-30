''' 

script for running ADCG on an image for source localization 

'''
import os
import sys 
import time 
import numpy as onp
import scipy.stats as stats
from PIL import Image

import autograd as Agrad
import autograd.numpy as np
import scipy.optimize

#####################################################################
# 
# >>> python run_adcg.py fimage sig_psf n_adcg foutput
# 
fimage  = sys.argv[1]           # image file name 
dataset = sys.argv[2]           # dataset 
n_adcg  = int(sys.argv[3])      # max iterations
foutput = sys.argv[4]           # output file name 
#####################################################################
background_subtract=False
#####################################################################

if dataset == 'BTLS': 
    sig_psf = 0.017133076722848264
else:
    raise NotImplementedError


def read_image(fframe, background_subtract=False):
    ''' read given image

    :param fframe: 
        file name. Currently assumes that the file can be opened using
        PIL.Image

    :param background_subtract:
        If True, sigmaclipped background subtraction (default: False) 
    '''
    im = Image.open(fframe)
    imarr = np.array(im)
    
    noise_level = 0. 
    if background_subtract: 
        # if true, simple sigmaclipped background subtraction
        noise_level = np.median(stats.sigmaclip(imarr.flatten(), high=3.)[0])
    return imarr - noise_level


image = read_image(fimage, background_subtract=background_subtract) # read image

# PSF 
cov_psf = sig_psf**2 * np.identity(2)
cinv_psf = np.linalg.inv(cov_psf)

Ngrid = image.shape[0] # grid size

pix = np.linspace(0., 1., Ngrid) # default pixel gridding 

xxpix, yypix = np.meshgrid(pix, pix) 
xypix = np.array([xxpix.flatten(), yypix.flatten()]).T


def psi(theta): 
    ''' measurement model (2d gaussian of width sigma PSF) written out to x,y grid
    '''
    return np.exp(-0.5 * np.array([dxy @ (cinv_psf @ dxy.T) for dxy in (xypix - theta[None,:])]))

theta_grid = np.linspace(0., 1., Ngrid) 
theta_xxgrid, theta_yygrid = np.meshgrid(theta_grid, theta_grid) 
theta_xygrid = np.array([theta_xxgrid.flatten(), theta_yygrid.flatten()]).T

def get_grid_psi(data_set='BTLS'): 
    fgridpsi = '%s_grid_psi.npy' % dataset.lower() 
    if os.path.isfile(fgridpsi): 
        return np.load(fgridpsi) 
    else: 
        grid_psi = np.stack([psi(tt) for tt in theta_xygrid])  
        np.save(fgridpsi, grid_psi) 
        return grid_psi

grid_psi = get_grid_psi(data_set=dataset)


def Psi(ws, thetas): 
    ''' "forward operator" i.e. forward model 
    
    Psi = int psi(theta) dmu(theta) 

    where mu is the signal parameter
    '''
    _thetas = np.atleast_2d(thetas)
    return np.sum(np.array([w * psi(tt) for (w,tt) in zip(ws, _thetas)]),0)


def ell(ws, thetas, yobs): 
    ''' loss function 
    '''
    if len(thetas.shape) == 1 and thetas.shape[0] > 2: 
        thetas = thetas.reshape((int(thetas.shape[0]/2), 2))
    return ((Psi(ws, thetas) - yobs)**2).sum() 


def gradell(ws, thetas, yobs):  
    ''' gradient of the loss fucntion 
    '''
    return (Psi(ws, thetas) - yobs)/((Psi(ws, thetas) - yobs)**2).sum() 


def lmo(v): 
    ''' step 1 of ADCG: "linear maximization oracle". This function does the following 
    optimization 
    
    argmin < psi(theta), v > 

    where for ADCG, v = the gradient of loss. For simplicity, we grid up theta to 
    theta_grid and calculate grid_psi minimize the inner product 
    '''
    ip = (grid_psi @ v) 
    return theta_xygrid[ip.argmin()] 


def coordinate_descent(thetas, yobs, lossFn, iter=35, min_drop=1e-5, **lossfn_kwargs):  
    ''' step 2 of ADCG (nonconvex optimization using block coordinate descent algorithm).
    compute weights, prune support, locally improve support
    '''
    def min_ws(): 
        # non-negative least square solver to find the weights that minimize loss 
        return scipy.optimize.nnls(np.stack([psi(tt) for tt in thetas]).T, yobs)[0]

    def min_thetas(): 
        res =  scipy.optimize.minimize(
                Agrad.value_and_grad(lambda tts: lossFn(ws, tts, yobs, **lossfn_kwargs)), thetas, 
                jac=True, method='L-BFGS-B', bounds=[(0.0, 1.0)]*2*thetas.shape[0])
        return res['x'], res['fun']

    old_f_val = np.Inf
    for i in range(iter): 
        thetas = np.atleast_2d(thetas)

        ws = min_ws() # get weights that minimize loss

        thetas, f_val = min_thetas() # keeping weights fixed, minimize loss 
    
        if len(thetas.shape) == 1 and thetas.shape[0] > 2: 
            thetas = thetas.reshape((int(thetas.shape[0]/2), 2))

        if old_f_val - f_val < min_drop: # if loss function doesn't improve by much
            break 
        old_f_val = f_val.copy()
    return ws, thetas 


def adcg(yobs, lossFn, gradlossFn, local_update, max_iters, **lossfn_kwargs): 
    ''' Alternative Descent Conditional Gradient 
    '''
    thetas, ws = np.zeros(0), np.zeros(0) 
    output = np.zeros(len(xypix)) 

    history = [] 
    for i in range(max_iters): 
        residual = output - yobs
        loss = lossFn(ws, thetas, yobs, **lossfn_kwargs) 
        print('  iter=%i, loss=%f' % (i, loss)) 
        history.append((loss, ws, thetas))
    
        # get gradient of loss function 
        grad = gradlossFn(ws, thetas, yobs, **lossfn_kwargs) 
        # compute new support
        theta = lmo(grad)
        # update signal parameters  
        if i == 0: _thetas = np.append(thetas, theta)
        else: _thetas = np.append(np.atleast_2d(thetas), np.atleast_2d(theta), axis=0)

        ws, thetas = local_update(_thetas, yobs, lossFn, **lossfn_kwargs)

        # calculate output 
        output = Psi(ws, thetas)
        
        if (i > 2) and (history[-2][0] - history[-1][0] < 1.):  
            return loss, ws, thetas

    return loss, ws, thetas

# run adcg 
loss, ws, thetas = adcg(
        image.flatten(), 
        ell, 
        gradell, 
        coordinate_descent, 
        n_adcg)

# writeout ADCG output  
np.savetxt(foutput, np.vstack([thetas.T, ws.T]).T) 
