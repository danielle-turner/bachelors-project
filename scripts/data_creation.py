import wigner_function as wf
import qutip as q
import numpy as np
from scipy.linalg import expm

split = 10
theta = np.linspace(0, np.pi / 2, split)
phi = np.linspace(0, np.pi, split)

i_sigma_z = np.matrix([[1j,   0],
                       [ 0, -1j]])
i_sigma_y = np.matrix([[ 0,   1],
                       [-1,   0]])
sigma_x   = np.matrix([[ 0,   1],
                       [ 1,   0]])
sigma_y   = np.matrix([[ 0, -1j],
                       [1j,   0]])
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
        rotation_operator[a,b] = np.matmul(expm(i_sigma_z * p),
                                  expm(i_sigma_y * t))

def rotation_gen():
    ra = 2*np.pi*np.asarray(np.random.rand(3,1), dtype=complex)
    a = np.matrix(np.asarray(i_sigma_z, dtype=complex)*ra[0])
    b = np.matrix(np.asarray(i_sigma_y, dtype=complex)*ra[1])
    c = np.matrix(np.asarray(i_sigma_z, dtype=complex)*ra[2])
    U = expm(a)*expm(b)*expm(c)
    return U

def rand_rotate(dm, n_qubits):
    U = rotation_gen()
    for i in range(n_qubits-1):
        Ui = rotation_gen()
        U = np.kron(U, Ui)
    U = np.matrix(U)
    return np.matmul(U, np.matmul(dm, U.getH()))

dms = []
wigners = []
n_qubits = 2
n_samples = 10000 # Number of samples for each type

for i in range(n_samples):
    # Random States
    dm = q.rand_dm(2**n_qubits, density=0.75, pure=True).full()
    if n_qubits == 2:
        wigner = wf.wigner_2qb_ea(dm, rotation_operator)
    elif n_qubits == 3:
        wigner = wf.wigner_3qb_ea(dm, rotation_operator)
    elif n_qubits == 4:
        wigner = wf.wigner_4qb_ea(dm, rotation_operator)
    elif n_qubits == 5:
        wigner = wf.wigner_5qb_ea(dm, rotation_operator)   
    wigners.append(wigner.flatten())
    # Random Separable States
    psi = np.matrix(q.rand_ket(2, density=0.75).full())
    for i in range(n_qubits-1):
        psi = np.kron(psi, np.matrix(q.rand_ket(2, density=0.75).full()))
    dm = psi*psi.getH()
    if n_qubits == 2:
        wigner = wf.wigner_2qb_ea(dm, rotation_operator)
    elif n_qubits == 3:
        wigner = wf.wigner_3qb_ea(dm, rotation_operator)
    elif n_qubits == 4:
        wigner = wf.wigner_4qb_ea(dm, rotation_operator)
    elif n_qubits == 5:
        wigner = wf.wigner_5qb_ea(dm, rotation_operator)   
    wigners.append(wigner.flatten())
    # Random Entangled States
    dm = np.zeros((2**n_qubits, 2**n_qubits))
    dm[0,0] = 0.5
    dm[0,2**n_qubits-1] = 0.5
    dm[2**n_qubits-1,0] = 0.5
    dm[2**n_qubits-1,2**n_qubits-1] = 0.5
    dm = rand_rotate(dm, n_qubits)
    if n_qubits == 2:
        wigner = wf.wigner_2qb_ea(dm, rotation_operator)
    elif n_qubits == 3:
        wigner = wf.wigner_3qb_ea(dm, rotation_operator)
    elif n_qubits == 4:
        wigner = wf.wigner_4qb_ea(dm, rotation_operator)
    elif n_qubits == 5:
        wigner = wf.wigner_5qb_ea(dm, rotation_operator)   
    wigners.append(wigner.flatten())
    
W = np.array(wigners)
np.save('wigner_functions_ea_{}'.format(n_qubits), W)
    
