import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import uuid
import os

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['cmr10']

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['cmr10']


def spre(op):
    return np.kron(op, np.eye(op.shape[0]))


def spost(op):
    return np.kron(np.eye(op.shape[0]), op.T)


def sprepost(A,B):
    #the Liouville space representation of A.R.B
    return np.kron(A, B.T)


def vec_lind(op):
    lio = 2 * sprepost(op, op.T.conj())
    lio = lio - spre(op.T.conj() @ op)
    lio = lio - spost(op.T.conj() @ op)
    return lio

def vec_trace(vec):
    dim = int(np.sqrt(vec.size))
    mat = vec.reshape((dim,dim), order='F')
    tr = np.trace(mat)
    return tr
    

def gaus(t, W, t0):
    gaus = (W**2/(2 * np.pi))**(0.25) * np.exp(-(0.5 * W * (t-t0))**2)
    return gaus

def zero(t):
    return 0
  
def dagger(vec):
    dim = int(np.sqrt(vec.size))
    mat = vec.reshape((dim,dim), order='F')
    mat = mat.T.conj()
    vec = mat.flatten('F')
    return vec





def prop_fock_state(t, y):

    """
    augmented_dens is a vector containing all the density matrices
    of interest.
    """
    # rho = np.array_splitaugmented_dens.reshape() 
    #system density matrices
    r00_t = y[0:36]
    r10_t = y[36:72]
    r01_t = y[72:108]
    r11_t = y[108:144]
    r20_t = y[144:180]
    r02_t = y[180:216]
    r21_t = y[216:252]
    r12_t = y[252:288]
    r22_t = y[288:324]


    
    #master equations
    r00_t_prop =  -1j * (spre(H_s) - spost(H_s)) @ r00_t
    r00_t_prop += Lind @ r00_t
    
    r10_t_prop =  -1j * (spre(H_s) - spost(H_s)) @ r10_t
    r10_t_prop += Lind @ r10_t 
    r10_t_prop += pulse_func(t) * (spost(L.T.conj()) - spre(L.T.conj())) @ r00_t

    r01_t_prop =  -1j * (spre(H_s) - spost(H_s)) @ r01_t
    r01_t_prop += Lind @ r01_t 
    r01_t_prop += np.conj(pulse_func(t)) * (spre(L) - spost(L)) @ r00_t

    r11_t_prop =  -1j * (spre(H_s) - spost(H_s)) @ r11_t
    r11_t_prop += Lind @ r11_t 
    r11_t_prop += pulse_func(t) * (spost(L.T.conj()) - spre(L.T.conj())) @ r01_t
    r11_t_prop += np.conj(pulse_func(t)) * (spre(L) - spost(L)) @ r10_t

    r20_t_prop =  -1j * (spre(H_s) - spost(H_s)) @ r20_t
    r20_t_prop += Lind @ r20_t 
    r20_t_prop += np.sqrt(2) * pulse_func(t) * (spost(L.T.conj()) - spre(L.T.conj())) @ r10_t

    r02_t_prop =  -1j * (spre(H_s) - spost(H_s)) @ r02_t
    r02_t_prop += Lind @ r02_t 
    r02_t_prop += np.sqrt(2) * np.conj(pulse_func(t)) * (spre(L) - spost(L)) @ r01_t

    r21_t_prop =  -1j * (spre(H_s) - spost(H_s)) @ r21_t
    r21_t_prop +=  Lind @ r21_t 
    r21_t_prop += np.sqrt(2) * pulse_func(t) * (spost(L.T.conj()) - spre(L.T.conj())) @ r11_t 
    r21_t_prop += np.conj(pulse_func(t)) * (spre(L) - spost(L)) @ r20_t

    r12_t_prop =  -1j * (spre(H_s) - spost(H_s)) @ r12_t
    r12_t_prop +=  Lind @ r12_t 
    r12_t_prop += pulse_func(t) * (spost(L.T.conj()) - spre(L.T.conj())) @ r02_t 
    r12_t_prop += np.sqrt(2) * np.conj(pulse_func(t)) * (spre(L) - spost(L)) @ r11_t

    r22_t_prop =  -1j * (spre(H_s) - spost(H_s)) @ r22_t
    r22_t_prop =  Lind @ r22_t 
    r22_t_prop += np.sqrt(2) * pulse_func(t) * (spost(L.T.conj()) - spre(L.T.conj())) @ r12_t
    r22_t_prop += np.sqrt(2) * np.conj(pulse_func(t)) * (spre(L) - spost(L)) @ r21_t

    #output photon flux expectation equations of motion
    Lambda00_t_prop =  vec_trace(spost(L.T.conj() @ L) @ dagger(r00_t))

    Lambda10_t_prop =  vec_trace(spost(L.T.conj() @ L) @ dagger(r10_t))
    Lambda10_t_prop += np.conj(pulse_func(t)) * vec_trace(spost(L) @ dagger(r00_t))

    Lambda01_t_prop =  vec_trace(spost(L.T.conj() @ L) @ dagger(r01_t))
    Lambda01_t_prop += pulse_func(t) * vec_trace(spost(L.T.conj()) @ dagger(r00_t))

    Lambda11_t_prop =  vec_trace(spost(L.T.conj() @ L) @ dagger(r11_t))
    Lambda11_t_prop += np.conj(pulse_func(t)) * vec_trace(spost(L) @ dagger(r01_t))
    Lambda11_t_prop += pulse_func(t) * vec_trace(spost(L.T.conj()) @ dagger(r10_t))
    Lambda11_t_prop += np.abs(pulse_func(t))**2 * vec_trace(dagger(r00_t))

    Lambda20_t_prop =  vec_trace(spost(L.T.conj() @ L) @ dagger(r20_t))
    Lambda20_t_prop += np.sqrt(2) * np.conj(pulse_func(t)) * vec_trace(spost(L) @ dagger(r10_t))

    Lambda02_t_prop =  vec_trace(spost(L.T.conj() @ L) @ dagger(r02_t))
    Lambda02_t_prop += np.sqrt(2) * pulse_func(t) * vec_trace(spost(L.T.conj()) @ dagger(r01_t))

    Lambda21_t_prop =  vec_trace(spost(L.T.conj() @ L) @ dagger(r21_t))
    Lambda21_t_prop += np.sqrt(2) * np.conj(pulse_func(t)) * vec_trace(spost(L) @ dagger(r11_t))
    Lambda21_t_prop += pulse_func(t) * vec_trace(spost(L.T.conj()) @ dagger(r20_t))
    Lambda21_t_prop += np.sqrt(2) * np.abs(pulse_func(t))**2 * vec_trace(dagger(r10_t))

    Lambda12_t_prop =  vec_trace(spost(L.T.conj() @ L) @ dagger(r12_t))
    Lambda12_t_prop += np.conj(pulse_func(t)) * vec_trace(spost(L) @ dagger(r02_t))
    Lambda12_t_prop += np.sqrt(2) * pulse_func(t) * vec_trace(spost(L.T.conj()) @ dagger(r11_t))
    Lambda12_t_prop += np.sqrt(2) * np.abs(pulse_func(t))**2 * vec_trace(dagger(r01_t))

    Lambda22_t_prop =  vec_trace(spost(L.T.conj() @ L) @ dagger(r22_t))
    Lambda22_t_prop += np.sqrt(2) * np.conj(pulse_func(t)) * vec_trace(spost(L) @ dagger(r12_t))
    Lambda22_t_prop += np.sqrt(2) * pulse_func(t) * vec_trace(spost(L.T.conj()) @ dagger(r21_t))
    Lambda22_t_prop += 2 * np.abs(pulse_func(t))**2 * vec_trace(dagger(r11_t))

    y = np.hstack((r00_t_prop, r10_t_prop, r01_t_prop, r11_t_prop, r20_t_prop, r02_t_prop, r21_t_prop, r12_t_prop, r22_t_prop, 
                  Lambda00_t_prop, Lambda10_t_prop, Lambda01_t_prop, Lambda11_t_prop, Lambda20_t_prop, Lambda02_t_prop, Lambda21_t_prop, Lambda12_t_prop, Lambda22_t_prop))

    return y

#Wavepacket
Omega = 1.46
t0 = 3
pulse_func = lambda x: gaus(x, Omega, t0)

#Field state(s)
#two photon
r_field_22 = np.array([[0,0,0],
                    [0,0,0],
                    [0,0,1]], dtype=complex)
r_field_22 = r_field_22.flatten('F')
#single photon
r_field_11 = np.array([[0,0,0],
                    [0,1,0],
                    [0,0,0]], dtype=complex)
r_field_11 = r_field_11.flatten('F')
#superposition of one and two
r_field_super = 0.5 * r_field_11 + 0.5 * r_field_22
initial_flux_expectations = np.zeros(r_field_11.size)



#Electronic state and operator definitions
psi_g = np.array([1,0], dtype=complex)
psi_e = np.array([0,1], dtype=complex)
sig = np.outer(psi_g, psi_e)
Gamma = 1
Lind = 0.5 * Gamma * vec_lind(np.kron(sig, np.eye(3)))
L = np.sqrt(Gamma) * np.kron(sig, np.eye(3))
S = np.kron(np.eye(2), np.eye(3))

#Vibrational state and operator definitions
phi_0 = np.array([1,0,0], dtype=complex)
phi_1 = np.array([0,1,0], dtype=complex)
phi_2 = np.array([0,0,1], dtype=complex)
a = np.array([[0,       np.sqrt(1),       0         ],
              [0,       0,                np.sqrt(2)],
              [0,       0,                0         ]], dtype=complex)
w_vib = 1
eta = 5
H_s_on = True
H_s =  H_s_on * w_vib * np.kron((a.T.conj() @ a), np.eye(2)) 
H_s += H_s_on * w_vib * eta * np.kron((sig.T.conj() @ sig), (a.T.conj() + a))

#Initial electronic state
r_el = 1 * np.outer(psi_g, psi_g) +  0 * np.outer(psi_e, psi_e) +  0 * np.outer(psi_e, psi_g) +  0 * np.outer(psi_g, psi_e)

#Initial vibrational state
r_vib = np.array([[1, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]], dtype=complex)


#Total system initial state
r_sys = np.kron(r_el, r_vib)
r_sys = r_sys.flatten('F')

r00_t0 = r_sys
r11_t0 = r_sys 
r22_t0 = r_sys 
r10_t0 = np.zeros(np.size(r_sys), dtype=complex)
r01_t0 = np.zeros(np.size(r_sys), dtype=complex)
r20_t0 = np.zeros(np.size(r_sys), dtype=complex)
r02_t0 = np.zeros(np.size(r_sys), dtype=complex)
r21_t0 = np.zeros(np.size(r_sys), dtype=complex)
r12_t0 = np.zeros(np.size(r_sys), dtype=complex)
initial_fock_dens = np.hstack((r00_t0, r10_t0, r01_t0, r11_t0, r20_t0, r02_t0, r21_t0, r12_t0, r22_t0))


#Time settings
tmax = 8
trange = np.linspace(0, tmax, 100)

#Solver
initial_conditions = np.hstack((initial_fock_dens, initial_flux_expectations))
sol = sc.integrate.solve_ivp(prop_fock_state, [0,tmax], initial_conditions, t_eval=trange,max_step=0.05)
print("sol size", np.shape(sol.y))
#Final states
r00_t = sol.y[0:36]
r10_t = sol.y[36:72]
r01_t = sol.y[72:108]
r11_t = sol.y[108:144]
r20_t = sol.y[144:180]
r02_t = sol.y[180:216]
r21_t = sol.y[216:252]
r12_t = sol.y[252:288]
r22_t = sol.y[288:324]

r_mn_t = np.array([r00_t,r01_t,r02_t,r10_t,r11_t,r12_t,r20_t,r21_t,r22_t])

#Final flux expectations
Lambda00_t = sol.y[324]
Lambda10_t = sol.y[325]
Lambda01_t = sol.y[326]
Lambda11_t = sol.y[327]
Lambda20_t = sol.y[328]
Lambda02_t = sol.y[329]
Lambda21_t = sol.y[330]
Lambda12_t = sol.y[331]
Lambda22_t = sol.y[332]

Lambda_mn_t = np.array([Lambda00_t,Lambda01_t,Lambda02_t,Lambda10_t,Lambda11_t,Lambda12_t,Lambda20_t,Lambda21_t,Lambda22_t])

#Total states
r_mn_weighted_t = np.zeros(r_mn_t.shape, dtype=complex)

for i in range(0,r_field_22.size) :
  r_mn_weighted_t[i] = r_field_22[i] * r_mn_t[i]
r_total_t_22 = np.sum(r_mn_weighted_t, axis=0)

for i in range(0,r_field_11.size) :
  r_mn_weighted_t[i] = r_field_11[i] * r_mn_t[i]
r_total_t_11 = np.sum(r_mn_weighted_t, axis=0)

for i in range(0,r_field_super.size) :
  r_mn_weighted_t[i] = r_field_super[i] * r_mn_t[i]
r_total_t_super = np.sum(r_mn_weighted_t, axis=0)

#Total fluxes
Lambda_mn_weighted_t = np.zeros(Lambda_mn_t.shape, dtype=complex)
for i in range(0,r_field_22.size) :
  Lambda_mn_weighted_t[i] = r_field_22[i] * Lambda_mn_t[i]
Lambda_total_t_22 = np.sum(Lambda_mn_weighted_t, axis=0)


for i in range(0,r_field_11.size) :
  Lambda_mn_weighted_t[i] = r_field_11[i] * Lambda_mn_t[i]
Lambda_total_t_11 = np.sum(Lambda_mn_weighted_t, axis=0)


for i in range(0,r_field_super.size) :
  Lambda_mn_weighted_t[i] = r_field_super[i] * Lambda_mn_t[i]
Lambda_total_t_super = np.sum(Lambda_mn_weighted_t, axis=0)

#flux
flux_22 = np.diff(Lambda_total_t_22)/np.diff(trange)
flux_11 = np.diff(Lambda_total_t_11)/np.diff(trange)
flux_00 = np.diff(Lambda00_t)/np.diff(trange)
flux_super = np.diff(Lambda_total_t_super)/np.diff(trange)

print("Flux_11: ")
print(" ")
print(flux_11)




#Probabilities
Pgg_22 = r_total_t_22[0] +  r_total_t_22[7] + r_total_t_22[14] 
Pee_22 = r_total_t_22[21] + r_total_t_22[28] + r_total_t_22[35] 

Pgg_11 = r_total_t_11[0] +  r_total_t_11[7] + r_total_t_11[14] 
Pee_11 = r_total_t_11[21] + r_total_t_11[28] + r_total_t_11[35] 


np.save("flux_11_eta_5_h_1.npy", flux_11)