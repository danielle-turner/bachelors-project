# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 05:39:22 2018

@author: User
"""

import tsne_V2 as tsne
import numpy as np
import matplotlib.pyplot as plt
import qutip as q
import datasheet as ds
import wigner_function as wf

split = 10

tolerance = 1e-5
theta = np.linspace(0, np.pi / 2, split)
phi = np.linspace(0, np.pi, split)

rotation_operator = np.zeros([split, split, 2, 2])
a = -1

for t in theta:
    b = -1
    a += 1
    for p in phi:
        b += 1
        rotation_operator[a,b] = (np.exp(ds.i_sigma_z * p) *
                                  np.exp(ds.i_sigma_y * t))

def rotation_operator_2qb(theta1, phi1, theta2, phi2):
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

def rotation_operator_3qb(theta1, phi1, theta2, phi2, theta3, phi3):
    #
    # Find the rotation operator for 3 qubits as defined in Eq. (5) of
    # 'Simple procedure for phase-space measurement and entanglement 
    # validation' by R. P. Rundle et al.
    #
    # Inputs :
    #       theta_i : Current theta for each qubit, denoted i.
    #       phi_i   : Current phi for each qubit, denoted i.
    #
    U1 = rotation_operator[theta1, phi1]
    U2 = rotation_operator[theta2, phi2]
    U3 = rotation_operator[theta3, phi3]
    Un = np.kron(U1, U2)
    Un = np.kron(Un, U3)
    return np.matrix(Un)

def rotation_operator_4qb(theta1, phi1, theta2, phi2, theta3, phi3,
                          theta4, phi4):
    #
    # Find the rotation operator for 4 qubits as defined in Eq. (5) of
    # 'Simple procedure for phase-space measurement and entanglement 
    # validation' by R. P. Rundle et al.
    #
    # Inputs :
    #       theta_i : Current theta for each qubit, denoted i.
    #       phi_i   : Current phi for each qubit, denoted i.
    #
    U1 = rotation_operator[theta1, phi1]
    U2 = rotation_operator[theta2, phi2]
    U3 = rotation_operator[theta3, phi3]
    U4 = rotation_operator[theta4, phi4]
    Un = np.kron(U1, U2)
    Un = np.kron(Un, U3)
    Un = np.kron(Un, U4)
    return np.matrix(Un)

def rotation_operator_5qb(theta1, phi1, theta2, phi2, theta3, phi3,
                          theta4, phi4, theta5, phi5):
    #
    # Find the rotation operator for 5 qubits as defined in Eq. (5) of
    # 'Simple procedure for phase-space measurement and entanglement 
    # validation' by R. P. Rundle et al.
    #
    # Inputs :
    #       theta_i : Current theta for each qubit, denoted i.
    #       phi_i   : Current phi for each qubit, denoted i.
    #
    U1 = rotation_operator[theta1, phi1]
    U2 = rotation_operator[theta2, phi2]
    U3 = rotation_operator[theta3, phi3]
    U4 = rotation_operator[theta4, phi4]
    U5 = rotation_operator[theta5, phi5]
    Un = np.kron(U1, U2)
    Un = np.kron(Un, U3)
    Un = np.kron(Un, U4)
    Un = np.kron(Un, U5)
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

def spin_u_density_matrix(N):
    #
    # Find the density matrix for N spin up qubits.
    #
    # Inputs :
    #       N : Number of qubits.
    #
    density_matrix = np.zeros((2**N, 2**N))
    density_matrix[0,0] = 1
    return np.matrix(density_matrix)

def wigner_function_2qb(density_matrix, t1, p1, t2, p2):
    #
    # Determine the Wigner function of 2 qubits using the given density matrix,
    # where the Wigner function is defined as in Eq. (12) of 'Simple procedure
    # for phase-space measurement and entanglement validation' by R. P. Rundle
    # et al.
    #
    ext_parity_2 = extended_parity(2)
    U2 = rotation_operator_2qb(t1, p1, t2, p2)
    U2_dagger = U2.getH()
    return np.trace(density_matrix * U2 * ext_parity_2 * U2_dagger)

def wigner_zeros_2qb(density_matrix):
    #
    # Determine the zeros of the Wigner function of 2 qubits using the given
    # density matrix, where the Wigner function is defined as in Eq. (12) of
    # 'Simple procedure for phase-space measurement and entanglement
    # validation' by R. P. Rundle et al.
    #
    x = range(split)
    W = []
    for t1, p1, t2, p2 in [(t1, p1, t2, p2) for t1 in x for p1 in x for t2 in x
                           for p2 in x]:
        result = wigner_function_2qb(density_matrix, t1, p1, t2, p2)
        if -tolerance < result < tolerance:
            W.append([theta[t1], phi[p1],
                      theta[t2], phi[p2]])
    return np.array(W)

def wigner_2qb_full(density_matrix):
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

def wigner_function_3qb(density_matrix, t1, p1, t2, p2, t3, p3):
    #
    # Determine the Wigner function of 3 qubits using the given density matrix,
    # where the Wigner function is defined as in Eq. (12) of 'Simple procedure
    # for phase-space measurement and entanglement validation' by R. P. Rundle
    # et al.
    #
    ext_parity_3 = extended_parity(3)
    U3 = rotation_operator_3qb(t1, p1, t2, p2, t3, p3)
    U3_dagger = U3.getH()
    return np.trace(density_matrix * U3 * ext_parity_3 * U3_dagger)

def wigner_zeros_3qb(density_matrix):
    #
    # Determine the zeros of the Wigner function of 3 qubits using the given
    # density matrix, where the Wigner function is defined as in Eq. (12) of
    # 'Simple procedure for phase-space measurement and entanglement
    # validation' by R. P. Rundle et al.
    #
    xt = range(split_theta)
    xp = range(split_phi)
    W = []
    for t1, p1, t2, p2, t3, p3 in [(t1, p1, t2, p2, t3, p3) for t1 in xt
                                   for p1 in xp for t2 in xt for p2 in xp
                                   for t3 in xt for p3 in xp]:
        result = wigner_function_3qb(density_matrix, t1, p1, t2, p2, t3, p3)
        if -tolerance < result < tolerance:
            W.append([theta_ghz[t1], phi_ghz[p1],
                      theta_ghz[t2], phi_ghz[p2],
                      theta_ghz[t3], phi_ghz[p3]])
    return np.array(W)

def wigner_function_4qb(density_matrix, t1, p1, t2, p2, t3, p3, t4, p4):
    #
    # Determine the Wigner function of 4 qubits using the given density matrix,
    # where the Wigner function is defined as in Eq. (12) of 'Simple procedure
    # for phase-space measurement and entanglement validation' by R. P. Rundle
    # et al.
    #
    ext_parity_4 = extended_parity(4)
    U4 = rotation_operator_4qb(t1, p1, t2, p2, t3, p3, t4, p4)
    U4_dagger = U4.getH()
    return np.trace(density_matrix * U4 * ext_parity_4 * U4_dagger)

def wigner_zeros_4qb(density_matrix):
    #
    # Determine the zeros of the Wigner function of 4 qubits using the given
    # density matrix, where the Wigner function is defined as in Eq. (12) of
    # 'Simple procedure for phase-space measurement and entanglement
    # validation' by R. P. Rundle et al.
    #
    x = range(split)
    W = []
    for t1, p1, t2, p2, t3, p3, t4, p4 in [(t1, p1, t2, p2, t3, p3, t4, p4)
        for t1 in x for p1 in x for t2 in x for p2 in x for t3 in x for p3 in x
        for t4 in x for p4 in x]:
        result = wigner_function_4qb(density_matrix, t1, p1, t2, p2, t3, p3,
                                     t4, p4)
        if -tolerance < result < tolerance:
            W.append([theta[t1], phi[p1],
                      theta[t2], phi[p2],
                      theta[t3], phi[p3],
                      theta[t4], phi[p4]])
    return np.array(W)

def wigner_function_5qb(density_matrix, t1, p1, t2, p2, t3, p3, t4, p4,
                        t5, p5):
    #
    # Determine the Wigner function of 5 qubits using the given density matrix,
    # where the Wigner function is defined as in Eq. (12) of 'Simple procedure
    # for phase-space measurement and entanglement validation' by R. P. Rundle
    # et al.
    #
    ext_parity_5 = extended_parity(5)
    U5 = rotation_operator_5qb(t1, p1, t2, p2, t3, p3, t4, p4, t5, p5)
    U5_dagger = U5.getH()
    return np.trace(density_matrix * U5 * ext_parity_5 * U5_dagger)

def wigner_zeros_5qb(density_matrix):
    #
    # Determine the zeros of the Wigner function of 5 qubits using the given
    # density matrix, where the Wigner function is defined as in Eq. (12) of
    # 'Simple procedure for phase-space measurement and entanglement
    # validation' by R. P. Rundle et al.
    #
    x = range(split)
    W = []
    for t1, p1, t2, p2, t3, p3, t4, p4, t5, p5 in [(t1, p1, t2, p2, t3, p3, t4,
        p4, t5, p5) for t1 in x for p1 in x for t2 in x for p2 in x for t3 in x
        for p3 in x for t4 in x for p4 in x for t5 in x for p5 in x]:
        result = wigner_function_5qb(density_matrix, t1, p1, t2, p2, t3, p3,
                                     t4, p4, t5, p5)
        if -tolerance < result < tolerance:
            W.append([theta[t1], phi[p1],
                      theta[t2], phi[p2],
                      theta[t3], phi[p3],
                      theta[t4], phi[p4],
                      theta[t5], phi[p5]])
    return W

dm = np.asarray(q.rand_dm(4, pure=False).full())
print(dm)

W = wigner_zeros_2qb(dm)

Wigner = tsne.tsne(W, perplexity=50.0, tolerance=1e-5, tries=50,
                   dimensions=2, learn_rate=200, max_iters=1000)
plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
plt.show()
plt.savefig('wigner_rand_mixed2_p50_lr200.png')
plt.clf()

dm = np.asarray(q.rand_dm(4, pure=False).full())
print(dm)

W = wigner_zeros_2qb(dm)

Wigner = tsne.tsne(W, perplexity=50.0, tolerance=1e-5, tries=50,
                   dimensions=2, learn_rate=200, max_iters=1000)
plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
plt.show()
plt.savefig('wigner_rand_mixed3_p50_lr200.png')
plt.clf()

dm = np.asarray(q.rand_dm(4, pure=False).full())
print(dm)

W = wigner_zeros_2qb(dm)

Wigner = tsne.tsne(W, perplexity=50.0, tolerance=1e-5, tries=50,
                   dimensions=2, learn_rate=200, max_iters=1000)
plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
plt.show()
plt.savefig('wigner_rand_mixed4_p50_lr200.png')
plt.clf()

dm = np.asarray(q.rand_dm(4, pure=True).full())
print(dm)

W = wigner_zeros_2qb(dm)

Wigner = tsne.tsne(W, perplexity=50.0, tolerance=1e-5, tries=50,
                   dimensions=2, learn_rate=200, max_iters=1000)
plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
plt.show()
plt.savefig('wigner_rand_pure1_p50_lr200.png')
plt.clf()

dm = np.asarray(q.rand_dm(4, pure=True).full())
print(dm)

W = wigner_zeros_2qb(dm)

Wigner = tsne.tsne(W, perplexity=50.0, tolerance=1e-5, tries=50,
                   dimensions=2, learn_rate=200, max_iters=1000)
plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
plt.show()
plt.savefig('wigner_rand_pure2_p50_lr200.png')
plt.clf()

dm = np.asarray(q.rand_dm(4, pure=True).full())
print(dm)

W = wigner_zeros_2qb(dm)

Wigner = tsne.tsne(W, perplexity=50.0, tolerance=1e-5, tries=50,
                   dimensions=2, learn_rate=200, max_iters=1000)
plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
plt.show()
plt.savefig('wigner_rand_pure3_p50_lr200.png')
plt.clf()

dm = np.asarray(q.rand_dm(4, pure=True).full())
print(dm)

W = wigner_zeros_2qb(dm)

Wigner = tsne.tsne(W, perplexity=50.0, tolerance=1e-5, tries=50,
                   dimensions=2, learn_rate=200, max_iters=1000)
plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
plt.show()
plt.savefig('wigner_rand_pure4_p50_lr200.png')
plt.clf()

#W = wigner_2qb_full(spin_du)
#np.save('wigner_spin_state_du_all_10split', W)

#W = np.load('wigner_ghz.npy')
#
#Wigner = tsne.tsne(W, perplexity=5.0, tolerance=1e-5, tries=50,
#                   dimensions=2, learn_rate=50, max_iters=1000,
#                   gamma=5)
#plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
#plt.show()
#plt.savefig('wigner_ghz_own_p5_lr50.png')
#plt.clf()
#
#Wigner = tsne.tsne(W, perplexity=30.0, tolerance=1e-5, tries=50,
#                   dimensions=2, learn_rate=50, max_iters=1000,
#                   gamma=5)
#plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
#plt.show()
#plt.savefig('wigner_ghz_own_p30_lr50.png')
#plt.clf()
#
#Wigner = tsne.tsne(W, perplexity=50.0, tolerance=1e-5, tries=50,
#                   dimensions=2, learn_rate=50, max_iters=1000,
#                   gamma=5)
#plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
#plt.show()
#plt.savefig('wigner_ghz_own_p50_lr50.png')
#plt.clf()
#
#Wigner = tsne.tsne(W, perplexity=100.0, tolerance=1e-5, tries=50,
#                   dimensions=2, learn_rate=50, max_iters=1000,
#                   gamma=5)
#plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
#plt.show()
#plt.savefig('wigner_ghz_own_p100_lr50.png')
#plt.clf()
#
#Wigner = tsne.tsne(W, perplexity=5.0, tolerance=1e-5, tries=50,
#                   dimensions=2, learn_rate=200, max_iters=1000,
#                   gamma=5)
#plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
#plt.show()
#plt.savefig('wigner_ghz_own_p5_lr200.png')
#plt.clf()
#
#Wigner = tsne.tsne(W, perplexity=30.0, tolerance=1e-5, tries=50,
#                   dimensions=2, learn_rate=200, max_iters=1000,
#                   gamma=5)
#plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
#plt.show()
#plt.savefig('wigner_ghz_own_p30_lr200.png')
#plt.clf()
#
#Wigner = tsne.tsne(W, perplexity=50.0, tolerance=1e-5, tries=50,
#                   dimensions=2, learn_rate=200, max_iters=1000,
#                   gamma=5)
#plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
#plt.show()
#plt.savefig('wigner_ghz_own_p50_lr200.png')
#plt.clf()
#
#Wigner = tsne.tsne(W, perplexity=100.0, tolerance=1e-5, tries=50,
#                   dimensions=2, learn_rate=200, max_iters=1000,
#                   gamma=5)
#plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
#plt.show()
#plt.savefig('wigner_ghz_own_p100_lr200.png')
#plt.clf()
#
#Wigner = tsne.tsne(W, perplexity=5.0, tolerance=1e-5, tries=50,
#                   dimensions=2, learn_rate=500, max_iters=1000,
#                   gamma=5)
#plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
#plt.show()
#plt.savefig('wigner_ghz_own_p5_lr500.png')
#plt.clf()
#
#Wigner = tsne.tsne(W, perplexity=30.0, tolerance=1e-5, tries=50,
#                   dimensions=2, learn_rate=500, max_iters=1000,
#                   gamma=5)
#plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
#plt.show()
#plt.savefig('wigner_ghz_own_p30_lr500.png')
#plt.clf()
#
#Wigner = tsne.tsne(W, perplexity=50.0, tolerance=1e-5, tries=50,
#                   dimensions=2, learn_rate=500, max_iters=1000,
#                   gamma=5)
#plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
#plt.show()
#plt.savefig('wigner_ghz_own_p50_lr500.png')
#plt.clf()
#
#Wigner = tsne.tsne(W, perplexity=100.0, tolerance=1e-5, tries=50,
#                   dimensions=2, learn_rate=500, max_iters=1000,
#                   gamma=5)
#plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
#plt.show()
#plt.savefig('wigner_ghz_own_p100_lr500.png')
#plt.clf()
#
#Wigner = tsne.tsne(W, perplexity=5.0, tolerance=1e-5, tries=50,
#                   dimensions=2, learn_rate=1000, max_iters=1000,
#                   gamma=5)
#plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
#plt.show()
#plt.savefig('wigner_ghz_own_p5_lr1000.png')
#plt.clf()
#
#Wigner = tsne.tsne(W, perplexity=30.0, tolerance=1e-5, tries=50,
#                   dimensions=2, learn_rate=1000, max_iters=1000,
#                   gamma=5)
#plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
#plt.show()
#plt.savefig('wigner_ghz_own_p30_lr1000.png')
#plt.clf()
#
#Wigner = tsne.tsne(W, perplexity=50.0, tolerance=1e-5, tries=50,
#                   dimensions=2, learn_rate=1000, max_iters=1000,
#                   gamma=5)
#plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
#plt.show()
#plt.savefig('wigner_ghz_own_p50_lr1000.png')
#plt.clf()
#
#Wigner = tsne.tsne(W, perplexity=100.0, tolerance=1e-5, tries=50,
#                   dimensions=2, learn_rate=1000, max_iters=1000,
#                   gamma=5)
#plt.scatter(Wigner[0][:, 0], Wigner[0][:, 1], c="r")
#plt.show()
#plt.savefig('wigner_ghz_own_p100_lr1000.png')
#plt.clf()