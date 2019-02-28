# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 19:48:14 2019

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Values = np.load('wigner_bell_00.npy')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Values[:190,1], Values[:190,2], Values[:190,3])
plt.savefig('wigner_bell_00_A_2D.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Values[190:227,1], Values[190:227,2], Values[190:227,3])
plt.savefig('wigner_bell_00_B_2D.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Values[227:264,1], Values[227:264,2], Values[227:264,3])
plt.savefig('wigner_bell_00_C_2D.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Values[264:301,1], Values[264:301,2], Values[264:301,3])
plt.savefig('wigner_bell_00_D_2D.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Values[301:338,1], Values[301:338,2], Values[301:338,3])
plt.savefig('wigner_bell_00_E_2D.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Values[338:375,1], Values[338:375,2], Values[338:375,3])
plt.savefig('wigner_bell_00_F_2D.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Values[375:412,1], Values[375:412,2], Values[375:412,3])
plt.savefig('wigner_bell_00_G_2D.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Values[412:449,1], Values[412:449,2], Values[412:449,3])
plt.savefig('wigner_bell_00_H_2D.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Values[449:486,1], Values[449:486,2], Values[449:486,3])
plt.savefig('wigner_bell_00_I_2D.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Values[486:,1], Values[486:,2], Values[486:,3])
plt.savefig('wigner_bell_00_J_2D.png')