import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import sklearn.metrics as met


def roll_fft(f):
    r,c = np.shape(f);
    f2 = np.roll(f,(c//2));
    f3 = np.roll(f2,(r//2),axis=-2);
    return f3;

def clip_sb(f):
    std = np.std(f[f<np.average(f)+3*np.std(f)]);
    avg = np.average(f[f<np.average(f)+3*np.std(f)]);
    return np.where(f<avg + 2*std,0,f+100)

def comp_jac(g,a,s):
    #make all nonzero into 1 and then give a score (since it's hard to get a point)
    aunit = np.where(0<a,1,0).astype(int);
    gunit = np.where(0<g,1,0).astype(int);
    sunit = np.where(0<s,1,0).astype(int);
    ajac = met.jaccard_score(gunit,aunit,average=None);
    sjac = met.jaccard_score(gunit,sunit)
    '''
    #normalize the peak to 1, each gets a soure intensity for free
    anorm = a/np.max(a);
    gnorm = g/np.max(g);
    snorm = s/np.max(s);
    ajac = met.jaccard_score(gnorm,anorm,average=None);
    sjac = met.jaccard_score(gnorm,snorm,average=None)
    '''
    return sjac,ajac;
    

imdir = '/home/moss/SMLM/data/sequence/';
fnum = 25

jacs = np.zeros((fnum,2));
for i in range(1,fnum):
    imageId = str(i+1).zfill(5);
    #sbpath = './local_out/'+imageId+'.out';
    sbpath ='./local_test/'+imageId+'.out';
    #sbpath ='./real'+imageId+'.out';
    adcgpath = './adcg_out/'+imageId+'.tiff';
    #nofftpath = './no_fft_out/' + imageId+'.out';
    grndpath = '/home/moss/SMLM/data/fluorophores/frames/'+imageId+'.csv';
    impath = imdir + imageId + '.tif';
    im = plt.imread(impath);
    sb = np.loadtxt(sbpath);
    #nofft = np.loadtxt(nofftpath);
    adcg = plt.imread(adcgpath);
    grnd = Table.read(grndpath, format='ascii');
    grnd_x = np.around(grnd['xnano']/100).astype(int);
    grnd_y = np.around(grnd['ynano']/100).astype(int);
    grnd_arr = 0*sb;
    grnd_arr[grnd_y,grnd_x] = grnd['intensity'];
    sb = roll_fft(sb);
    sb_p = clip_sb(sb);
    #jacs[i,:] = comp_jac(grnd_arr,adcg,sb);    
        
    
    
    #figpath = './compare_out/'+imageId+'.png';
    figpath = './'+imageId+'.png';
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(sb);
    ax[0].scatter(grnd_x,grnd_y,s=80, facecolors='none', edgecolors='r')
    ax[0].set_title('Sparse Bayes')
    ax[1].imshow(sb_p);
    ax[1].scatter(grnd_x,grnd_y,s=80, facecolors='none', edgecolors='r')
    ax[1].set_title('Sparse Bayes +100')
    ax[2].imshow(adcg);
    ax[2].scatter(grnd_x,grnd_y,s=80, facecolors='none', edgecolors='r')
    ax[2].set_title('ADCG')
    #ax[2].imshow(nofft)
    #ax[2].scatter(grnd_x,grnd_y,s=80, facecolors='none', edgecolors='r')
    #ax[2].set_title('No FFFT')
    fig.savefig(figpath,dpi=300);
    #plt.close();
    plt.show();
    
    
#plt.plot(jacs[:,0],label='Sparse Bayes');
#plt.plot(jacs[:,1],label='ADCG')
#plt.legend();



