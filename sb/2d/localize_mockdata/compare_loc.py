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
    std = np.std(f)#np.std(f[f<np.average(f)+3*np.std(f)]);
    avg = np.average(f)#np.average(f[f<np.average(f)+3*np.std(f)]);
    print(avg);
    print(std);
    return np.where(f<0.5,0,f)#return np.where(f<avg + 6*std,0,f)
    

fnum = 25
dataset = 'hd_lsnr'
jacs = np.zeros((fnum,2));
for i in range(0,fnum):
    impath = '/home/moss/Projects/LocaHBay/sb/2d/data/' +dataset +'/'+ str(i) +'.dat'
    svpath = '/home/moss/Projects/LocaHBay/sb/2d/localize_mockdata/' +dataset+'/'+ str(i) +'.out'
    grndpath = '/home/moss/Projects/LocaHBay/sb/2d/data/'+dataset+'/'+str(i)+'.truth';
    im = np.loadtxt(impath);
    sb = np.loadtxt(svpath);
    grnd = np.loadtxt(grndpath)
    grnd = Table.read(grndpath,format='ascii')
    grnd_x = np.around(grnd['col1']).astype(int);
    grnd_y = np.around(grnd['col2']).astype(int);
    grnd_arr = 0*sb;
    grnd_arr[grnd_x,grnd_y] = grnd['col3'];
    sb = roll_fft(sb);
    sb = clip_sb(sb);    
        
    
    
    #figpath = './compare_out/'+imageId+'.png';
    figpath = './compare_'+dataset+'/'+str(i)+'.png';
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(sb);
    ax[0].scatter(grnd_y,grnd_x,s=80, facecolors='none', edgecolors='r')
    ax[0].set_title('Sparse Bayes')
    ax[1].imshow(im);
    ax[1].set_title('Data')
    #ax[2].imshow(adcg);
    #ax[2].scatter(grnd_x,grnd_y,s=80, facecolors='none', edgecolors='r')
    #ax[2].set_title('ADCG')
    fig.savefig(figpath,dpi=300);
    #plt.close();
    plt.show();
    
    
#plt.plot(jacs[:,0],label='Sparse Bayes');
#plt.plot(jacs[:,1],label='ADCG')
#plt.legend();



