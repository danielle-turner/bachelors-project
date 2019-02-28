# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:54:27 2019

@author: User
"""

from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt
import qutip as q

#import tsne_V2 as tsne

split = 10
theta = np.linspace(0, np.pi / 2, split)
phi = np.linspace(0, np.pi, split)
tolerance = 1e-5

i_sigma_z = np.matrix([[1j,   0],
                       [ 0, -1j]])
i_sigma_y = np.matrix([[ 0,   1],
                       [-1,   0]])
sigma_z   = np.matrix([[ 1,   0],
                       [ 0,  -1]])
identity  = np.matrix([[ 1,   0],
                       [ 0,   1]])

rotation_operator = np.zeros([split, split, 2, 2])
a = -1

for t in theta:
    b = -1
    a += 1
    for p in phi:
        b += 1
        rotation_operator[a,b] = (np.exp(i_sigma_z * p) *
                                  np.exp(i_sigma_y * t))

def rotation_operator_2qb(theta1, phi1, theta2, phi2, rotation_operator):
    #
    # Find the rotation operator for 2 qubits as defined in Eq. (5) of
    # 'Simple procedure for phase-space measurement and entanglement 
    # validation' by R. P. Rundle et al.
    #
    # Inputs :
    #       theta_i : Current theta for each qubit, denoted i.
    #       phi_i   : Current phi for each qubit, denoted i.
    #
    U1 = rotation_operator[theta1, phi1]
    U2 = rotation_operator[theta2, phi2]
    Un = np.kron(U1, U2)
    return np.matrix(Un)

def extended_parity(N):
    #
    # Find the extended parity operator for N qubits as defined in Eq. (10) of
    # 'Simple procedure for phase-space measurement and entanglement 
    # validation' by R. P. Rundle et al.
    #
    # Inputs :
    #       N : Number of qubits.
    #
    ext_parity_i = 0.5 * (identity + (np.sqrt(3) * sigma_z))
    ext_parity = ext_parity_i
    if N > 1:
        for n in range(N-1):
            ext_parity = np.kron(ext_parity, ext_parity_i)
    return ext_parity

def wigner_function_2qb(density_matrix, t1, p1, t2, p2):
    #
    # Determine the Wigner function of 2 qubits using the given density matrix,
    # where the Wigner function is defined as in Eq. (12) of 'Simple procedure
    # for phase-space measurement and entanglement validation' by R. P. Rundle
    # et al.
    #
    ext_parity_2 = extended_parity(2)
    U2 = rotation_operator_2qb(t1, p1, t2, p2, rotation_operator)
    U2_dagger = U2.getH()
    return np.trace(density_matrix * U2 * ext_parity_2 * U2_dagger)

def wigner_zeros_2qb(density_matrix, split, tolerance):
    #
    # Determine the zeros of the Wigner function of 2 qubits using the given
    # density matrix, where the Wigner function is defined as in Eq. (12) of
    # 'Simple procedure for phase-space measurement and entanglement
    # validation' by R. P. Rundle et al.
    #
    theta = np.linspace(0, np.pi / 2, split)
    phi = np.linspace(0, np.pi, split)
    x = range(split)
    W = []
    for t1, p1, t2, p2 in [(t1, p1, t2, p2) for t1 in x for p1 in x for t2 in x
                           for p2 in x]:
        result = wigner_function_2qb(density_matrix, t1, p1, t2, p2)
        if -tolerance < result < tolerance:
            W.append([theta[t1], phi[p1],
                      theta[t2], phi[p2]])
    return np.array(W)

def wigner_2qb_full(density_matrix, split):
    #
    # Determine the zeros of the Wigner function of 2 qubits using the given
    # density matrix, where the Wigner function is defined as in Eq. (12) of
    # 'Simple procedure for phase-space measurement and entanglement
    # validation' by R. P. Rundle et al.
    #
    x = range(split)
    W = np.zeros((split,split,split,split))
    for t1, p1, t2, p2 in [(t1, p1, t2, p2) for t1 in x for p1 in x for t2 in x
                           for p2 in x]:
        result = wigner_function_2qb(density_matrix, t1, p1, t2, p2)
        W[t1, p1][t2, p2] = result
    return np.array(W)

wigners = []
colours = []

random = True

if random:
    N = 100
    for i in range(int(N / 2)):
        dm = np.asarray(q.rand_dm(4, pure=False).full())
        wigner = wigner_2qb_full(dm, split).flatten()
        wigner.shape = (-1, 1)
        wigners.append(wigner)
        colours.append(np.trace(dm**2))
        
    for i in range(int(N / 2)):
        dm = np.asarray(q.rand_dm(4, pure=True).full())
        wigner = wigner_2qb_full(dm, split).flatten()
        wigner.shape = (-1, 1)
        wigners.append(wigner)
        colours.append(np.trace(dm**2))
        
else:
    dms = np.load('density_matrices.npy')
    for dm in dms:
        wigner = wigner_2qb_full(dm, split).flatten()
        wigner.shape = (-1, 1)
        wigners.append(wigner)
        colours.append(np.trace(dm**2))
    
X = np.asarray(wigners)[:, :, 0]

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
                     perplexity=5.0, learning_rate=50.0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=colours, cmap='RdYlBu')
plt.show()
plt.savefig('cluster_p5_lr50.png')
plt.clf()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
                     perplexity=30.0, learning_rate=50.0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=colours, cmap='RdYlBu')
plt.show()
plt.savefig('cluster_p30_lr50.png')
plt.clf()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
                     perplexity=50.0, learning_rate=50.0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=colours, cmap='RdYlBu')
plt.show()
plt.savefig('cluster_p50_lr50.png')
plt.clf()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
                     perplexity=100.0, learning_rate=50.0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=colours, cmap='RdYlBu')
plt.show()
plt.savefig('cluster_p100_lr50.png')
plt.clf()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
                     perplexity=5.0, learning_rate=200.0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=colours, cmap='RdYlBu')
plt.show()
plt.savefig('cluster_p5_lr200.png')
plt.clf()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
                     perplexity=30.0, learning_rate=200.0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=colours, cmap='RdYlBu')
plt.show()
plt.savefig('cluster_p30_lr200.png')
plt.clf()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
                     perplexity=50.0, learning_rate=200.0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=colours, cmap='RdYlBu')
plt.show()
plt.savefig('cluster_p50_lr200.png')
plt.clf()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
                     perplexity=100.0, learning_rate=200.0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=colours, cmap='RdYlBu')
plt.show()
plt.savefig('cluster_p100_lr200.png')
plt.clf()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
                     perplexity=5.0, learning_rate=500.0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=colours, cmap='RdYlBu')
plt.show()
plt.savefig('cluster_p5_lr500.png')
plt.clf()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
                     perplexity=30.0, learning_rate=500.0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=colours, cmap='RdYlBu')
plt.show()
plt.savefig('cluster_p30_lr500.png')
plt.clf()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
                     perplexity=50.0, learning_rate=500.0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=colours, cmap='RdYlBu')
plt.show()
plt.savefig('cluster_p50_lr500.png')
plt.clf()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
                     perplexity=100.0, learning_rate=500.0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=colours, cmap='RdYlBu')
plt.show()
plt.savefig('cluster_p100_lr500.png')
plt.clf()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
                     perplexity=5.0, learning_rate=1000.0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=colours, cmap='RdYlBu')
plt.show()
plt.savefig('cluster_p5_lr1000.png')
plt.clf()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
                     perplexity=30.0, learning_rate=1000.0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=colours, cmap='RdYlBu')
plt.show()
plt.savefig('cluster_p30_lr1000.png')
plt.clf()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
                     perplexity=50.0, learning_rate=1000.0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=colours, cmap='RdYlBu')
plt.show()
plt.savefig('cluster_p50_lr1000.png')
plt.clf()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
                     perplexity=100.0, learning_rate=1000.0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=colours, cmap='RdYlBu')
plt.show()
plt.savefig('cluster_p100_lr1000.png')
plt.clf()