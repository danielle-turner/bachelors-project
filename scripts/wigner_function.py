# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 08:51:02 2019

@author: User
"""
import numpy as np
import datasheet as ds


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

def rotation_operator_3qb(theta1, phi1, theta2, phi2, theta3, phi3,
                          rotation_operator):
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
                          theta4, phi4, rotation_operator):
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
                          theta4, phi4, theta5, phi5, rotation_operator):
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
    ext_parity_i = 0.5 * (ds.identity + (np.sqrt(3) * ds.sigma_z))
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
    U2 = rotation_operator_2qb(t1, p1, t2, p2)
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

def wigner_zeros_3qb(density_matrix, split, tolerance):
    #
    # Determine the zeros of the Wigner function of 3 qubits using the given
    # density matrix, where the Wigner function is defined as in Eq. (12) of
    # 'Simple procedure for phase-space measurement and entanglement
    # validation' by R. P. Rundle et al.
    #
    theta = np.linspace(0, np.pi / 2, split)
    phi = np.linspace(0, np.pi, split)
    x = range(split)
    W = []
    for t1, p1, t2, p2, t3, p3 in [(t1, p1, t2, p2, t3, p3) for t1 in x
                                   for p1 in x for t2 in x for p2 in x
                                   for t3 in x for p3 in x]:
        result = wigner_function_3qb(density_matrix, t1, p1, t2, p2, t3, p3)
        if -tolerance < result < tolerance:
            W.append([theta[t1], phi[p1],
                      theta[t2], phi[p2],
                      theta[t3], phi[p3]])
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

def wigner_zeros_4qb(density_matrix, split, tolerance):
    #
    # Determine the zeros of the Wigner function of 4 qubits using the given
    # density matrix, where the Wigner function is defined as in Eq. (12) of
    # 'Simple procedure for phase-space measurement and entanglement
    # validation' by R. P. Rundle et al.
    #
    theta = np.linspace(0, np.pi / 2, split)
    phi = np.linspace(0, np.pi, split)
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

def wigner_zeros_5qb(density_matrix, split, tolerance):
    #
    # Determine the zeros of the Wigner function of 5 qubits using the given
    # density matrix, where the Wigner function is defined as in Eq. (12) of
    # 'Simple procedure for phase-space measurement and entanglement
    # validation' by R. P. Rundle et al.
    #
    theta = np.linspace(0, np.pi / 2, split)
    phi = np.linspace(0, np.pi, split)
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