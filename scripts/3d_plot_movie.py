# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 13:15:14 2019

@author: phdct
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm, colors, colorbar

Values = np.load('wigner_bell_state_11_all.npy')
norm = colors.Normalize(vmin = np.min(Values),
                        vmax = np.max(Values), clip = False)
for i in range(30):
    for j in range(30):
        
        u, v = np.mgrid[0:np.pi:30j, 0:2*np.pi:30j]
        
        strength = Values[i, j]
        
        x = 10 * np.sin(u) * np.cos(v)
        y = 10 * np.sin(u) * np.sin(v)
        z = 10 * np.cos(u)
        
        im = plt.imshow(strength, cmap=plt.cm.RdBu, vmin = np.min(Values),
                        vmax = np.max(Values))
        plt.colorbar(im)
        plt.savefig('wigner_bell_11_' + str(i) + '_' + str(j) + '.png')
        plt.close()