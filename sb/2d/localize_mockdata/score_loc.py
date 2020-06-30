import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import sklearn as sk


class score_loc:
    
    def __init__(self,res, ground):
        self.res = res;
        self.ground = ground;
        self.truths = self.getPoints(ground);
        self.sources = self.getPoints(res);
        self.run()
    
    #returns indice of each nonzero point in arr    
    def getPoints(self,arr):
        return np.nonzero(arr);
    
    
    def run():
                
       
        
    
        