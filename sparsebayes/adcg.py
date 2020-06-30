'''

implementation of ADCG. Notation and implementation 
follow N. Boyd's ADCG paper.

'''
import numpy as np 
import scipy as sp 


def ell(Psi, ws, thetas, yobs): 
    ''' given forward operator, signal parameters, and observations calculate 
    loss function 
    '''
    return ((Psi(ws, thetas) - yobs)**2).sum() 


def lmo(g): 
    ''' "linear minimization oracle". This function does the following optimization 
    
    argmin < psi(theta), g > 

    where for ADCG, g = the gradient of loss. Since this is a 1D problem, 
    we grid up theta to theta_grid and calculate grid_psi minimize the 
    inner product 
    '''
    ip = np.matmul(grid_psi, g) 
    return theta_grid[ip.argmin()] 


def coordinate_descent(thetas, yobs, lossFn, iter=35, min_drop=1e-5, **lossfn_kwargs):  
    ''' block coordinate descent algorithm. 
    compute weights, prune support, locally improve support
    '''
    def min_ws(): 
        # non-negative least square solver to find the weights that minimize loss 
        return scipy.optimize.nnls(np.stack([psi(tt) for tt in thetas]).T, yobs)[0]
    def min_thetas(): 
        res =  scipy.optimize.minimize(
                Agrad.value_and_grad(lambda tts: lossFn(ws, tts, yobs, **lossfn_kwargs)), thetas, 
                jac=True, method='L-BFGS-B', bounds=[(0.0, 1.0)]*len(thetas))
        return res['x'], res['fun']
    
    old_f_val = np.Inf
    for i in range(iter): 
        ws = min_ws() # get weights that minimize loss
        thetas, f_val = min_thetas() # keeping weights fixed, minimize loss 
        if old_f_val - f_val < min_drop: # if loss function doesn't improve by much
            break 
        old_f_val = f_val 
    return ws, thetas 


def adcg(yobs, lossFn, local_update, max_iters, **lossfn_kwargs): 
    ''' Alternative Descent Conditional Gradient method implementation 
    carefully following Nick Boyd's tutorial. Mainly involves two optimization 
    problems
    '''
    thetas, ws = np.zeros(0), np.zeros(0) 
    output = np.zeros(len(xpix)) 
    history = [] 
    for i in range(max_iters): 
        residual = output - yobs
        #loss = (residual**2).sum() 
        loss = lossFn(ws, thetas, yobs, **lossfn_kwargs) 
        print('iter=%i, loss=%f' % (i, loss)) 
        history.append((loss, ws, thetas))

        theta = lmo(residual) 
        ws, thetas = local_update(np.append(thetas, theta), yobs, lossFn, **lossfn_kwargs)
        output = Psi(ws, thetas)
    return history 


def select_k(history): 
    drop = np.array([history[i][0]-history[i+1][0] for i in range(len(history)-1)])
    k_hat = np.argmax(drop<0.1)
    return history[k_hat]


def ADCG_1d(loss_str='l2', N_source=3, sig_noise=0.2, seed=1): 
    '''
    '''
    thetas_true, weights_true, yobs = obs1d(N_source=N_source, sig_noise=sig_noise, seed=seed)
    fdensity = float(N_source) / float(len(theta_grid)) # for now lets assume we know the density perfectly. 

    if loss_str is 'l2': 
        hist = adcg(yobs, ell, coordinate_descent, 30)
    elif loss_str == 'l2_noise':
        hist = adcg(yobs, ell_noise, coordinate_descent, 30)
    elif loss_str == 'l2_noise_sparse': 
        hist = adcg(yobs, ell_sparseprior, coordinate_descent, 30, fdensity=fdensity, sig_noise=sig_noise)
    loss, ws, thetas = select_k(hist)
    output = Psi(ws, thetas) 
    print ws 

    # plot data 
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.errorbar(xpix, yobs, yerr=sig_noise, elinewidth=0.2, fmt='xC0', zorder=1)
    for x, w in zip(thetas_true, weights_true):
        sub.vlines(x, 0, w, color='k', linewidth=1.5, linestyle='--')

    if len(ws) > 0:
        for x, w in zip(thetas, ws):
            _plt = sub.vlines(x, 0, w, color='C1')
        sub.legend([_plt], ['inf. intensities'], loc='upper right', fontsize=15) 
    sub.set_xlabel(r'$x_{\rm pix}$', fontsize=25) 
    sub.set_xlim(-0.1, 1.1) 
    sub.set_ylabel('intensity', fontsize=25) 
    sub.set_ylim(0., 2.5) 
    ffig = os.path.join(UT.fig_dir(), 
            'obs1d.psf%s.nois%s.Nsource%i.seed%i.%s.adcg.png' % (str(sig_psf), str(sig_noise), N_source, seed, loss_str))
    fig.savefig(ffig, 
            bbox_inches='tight')
    return None 



'''
    def GaussBlur2D(s, n_pixels, ng):  
        sigma = np.sqrt(s) 
        grid = np.linspace(0., 1., ng) 
        grid_f = computeFs(np.linspace(0., 1., ng), n_pixels, sigma)
        u = np.zeros(n_pixels) 
        l = np.zeros(n_pixels)  
        return None 


    def computeFs(x, n_pixels, sigma):
        ng = len(x)
        z = zeros(n_pixels)
        u = zeros(n_pixels)
        l = zeros(n_pixels) 

        result = np.zeros((n_pixels, ng))
        for i in xrange(ng): 
            z = _computeF(z, sigma, u, l, x[i])
            for j in xrange(n_pixels): 
                result[j,i] = z[j]
        return result


    def _computeF(f, sigma, u, l, x): 
        n_pixels = len(u)
        inc = 1.0/(sigma * n_pixels)
        xovers = x/sigma

        prefactor = sigma * np.sqrt(np.pi) / 2.0 
        for i in xrange(n_pixels):
            u[i] = (i+1) * inc - xovers
            l[i] = i * inc - xovers 
        for i in xrange(n_pixels):
            f[i] = prefactor * (sp.special.erf(u[i]) - sp.special.erf(l[i]))
        return f
'''

