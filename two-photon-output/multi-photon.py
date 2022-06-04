import numpy as np
import scipy as sc
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import uuid
import os
from warnings import warn

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['cmr10']

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['cmr10']

mpl.rcParams.update({'font.size': 14})







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
    mat = vec.reshape((dim,dim))
    tr = np.trace(mat)
    return tr
    

def gaus(t, W, t0):
    gaus = (W**2/(2 * np.pi))**(0.25) * np.exp(-(0.5 * W * (t-t0))**2)
    return gaus

def zero(t):
    return 0
  
def dagger(vec):
    dim = int(np.sqrt(vec.size))
    mat = vec.reshape((dim,dim))
    mat = mat.T.conj()
    vec = mat.reshape(-1)
    return vec


class system_matrix:
  '''
  System matrix with dimesnsions x,y,z 
  x,y specify reduced system density matrix for Fock state |x><y|
  z elements are system state matrix elements in order 00, 10, 01, 11, 20, 02, 21...
  '''
  def __init__(self, sys_matrix:np.ndarray=None, fock_size=2, sys_size=2) -> np.ndarray:
    if type(sys_matrix) is not np.ndarray:
      self.matrix = np.zeros((fock_size,fock_size,sys_size**2))
    else:
      self.matrix = sys_matrix

def prop_state(t, system_matrix):
  rho = system_matrix
  rho_prop = np.zeros(rho.shape)
  for m in range(0,rho.shape[0]):
    for n in range(0,rho.shape[1]):
      rho_prop[m][n] =   Lind @ rho[m][n]
      if m > 0:
        rho_prop[m][n] +=  np.sqrt(m) * pulse_func(t) * (spost(L.T.conj()) - spre(L.T.conj())) @ rho[m-1][n]
      if n > 0:
        rho_prop[m][n] +=  np.sqrt(n) * np.conj(pulse_func(t)) * (spre(L) - spost(L)) @ rho[m][n-1]

  return rho_prop

def prop_flux(t, system_matrix):
  rho = system_matrix
  flux = np.zeros(shape=(system_matrix.shape[0], system_matrix.shape[1]))
  for m in range(0,flux.shape[0]):
    for n in range(0,flux.shape[1]):
      flux[m,n] =  vec_trace(spost(L.T.conj() @ L) @ dagger(rho[m][n]))
      if m > 0:
        flux[m,n] += np.sqrt(m)   * np.conj(pulse_func(t)) * vec_trace(spost(L) @ dagger(rho[m-1][n]))
      if n > 0:
        flux[m,n] += np.sqrt(n)   * pulse_func(t) * vec_trace(spost(L.T.conj()) @ dagger(rho[m][n-1]))
      if (m > 0) and (n > 0):
        flux[m,n] += np.sqrt(m*n) * np.abs(pulse_func(t))**2 * vec_trace(dagger(rho[m-1][n-1]))
   
  return flux   

def prop_state_flux(t, y):
  rho = y[0:(fock_size**2 * sys_size**2)]
  rho = np.reshape(rho, (fock_size, fock_size, sys_size**2))
  
  rho_prop = prop_state(t, rho)
  flux_prop = prop_flux(t, rho)

  rho_prop = np.reshape(rho_prop, -1)
  flux_prop = np.reshape(flux_prop, -1)

  y = np.hstack((rho_prop, flux_prop))
  flag = True
  return y

def weight(r_field, r_mn_t):
  rho_mn_weighted_t = np.zeros(r_mn_t.shape)
  for m in range(0,r_field.shape[0]):
    for n in range(0,r_field.shape[1]): 
      rho_mn_weighted_t[m][n] = r_field[m][n] * r_mn_t[m][n]
  
  r_total_t = np.sum(rho_mn_weighted_t, axis=(0,1))
  return r_total_t 


#Wavepacket
Omega = 1.46
t0 = 3
pulse_func = lambda x: gaus(x, Omega, t0)

#Field state(s)
fock_size = 3
#two photon
r_field_22 = np.array([[0,0,0],
                    [0,0,0],
                    [0,0,1]])
#r_field_22 = r_field_22.ravel()
#single photon
r_field_11 = np.array([[0,0,0],
                    [0,1,0],
                    [0,0,0]])
#r_field_11 = r_field_11.ravel()
#superposition of one and two
r_field_super = 0.5 * r_field_11 + 0.5 * r_field_22
initial_flux_expectations = np.zeros(r_field_11.size)



#State and operator definitions
psi_e = np.array([1,0])
psi_g = np.array([0,1])
sig = np.outer(psi_g, psi_e)
Gamma = 1
Lind = 0.5 * Gamma * vec_lind(sig)
L = np.sqrt(Gamma) * sig
S = np.eye(2)

#Initial emitter state
sys_size = 2
r_sys = 1 * np.outer(psi_g, psi_g).reshape(-1) +  0 * np.outer(psi_e, psi_e).reshape(-1) +  0 * np.outer(psi_e, psi_g).reshape(-1) +  0 * np.outer(psi_g, psi_e).reshape(-1)
r00_t0 = r_sys
r11_t0 = r_sys 
r22_t0 = r_sys 
r10_t0 = np.zeros(4)
r01_t0 = np.zeros(4)
r20_t0 = np.zeros(4)
r02_t0 = np.zeros(4)
r21_t0 = np.zeros(4)
r12_t0 = np.zeros(4)
initial_fock_dens = np.hstack((r00_t0, r01_t0, r02_t0, r10_t0, r11_t0, r12_t0, r20_t0, r21_t0, r22_t0))

#Time settings
tmax = 8
trange = np.linspace(0, tmax, 100)

#Solver
initial_conditions = np.hstack((initial_fock_dens, initial_flux_expectations))
sol = sc.integrate.solve_ivp(prop_state_flux, [0,tmax], initial_conditions, t_eval=trange,max_step=0.05)

#Final states
rho_mn_t = sol.y[0:(fock_size**2 * sys_size**2)]
rho_mn_t = np.reshape(rho_mn_t, (fock_size, fock_size, sys_size**2, np.size(trange)))
r00_t = sol.y[0:4]
r01_t = sol.y[4:8]
r02_t = sol.y[8:12]
r10_t = sol.y[12:16]
r11_t = sol.y[16:20]
r12_t = sol.y[20:24]
r20_t = sol.y[24:28]
r21_t = sol.y[28:32]
r22_t = sol.y[32:36]

r_mn_tx = np.array([r00_t,r01_t,r02_t,r10_t,r11_t,r12_t,r20_t,r21_t,r22_t])




#Final flux expectations
Lambda_mn_t = sol.y[(fock_size**2 * sys_size**2):]
Lambda_mn_t = np.reshape(Lambda_mn_t, (fock_size, fock_size, np.size(trange)))
Lambda00_t = sol.y[36]
Lambda01_t = sol.y[37]
Lambda02_t = sol.y[38]
Lambda10_t = sol.y[39]
Lambda11_t = sol.y[40]
Lambda12_t = sol.y[41]
Lambda20_t = sol.y[42]
Lambda21_t = sol.y[43]
Lambda22_t = sol.y[44]

Lambda_mn_tx = np.array([Lambda00_t,Lambda01_t,Lambda02_t,Lambda10_t,Lambda11_t,Lambda12_t,Lambda20_t,Lambda21_t,Lambda22_t])





#Total states
r_total_t_22 = weight(r_field_22, rho_mn_t)
r_total_t_11 = weight(r_field_11, rho_mn_t)
r_total_t_super = weight(r_field_super, rho_mn_t)

#Total fluxes
Lambda_total_t_22 = weight(r_field_22, Lambda_mn_t)
Lambda_total_t_11 = weight(r_field_11, Lambda_mn_t)
Lambda_total_t_super = weight(r_field_super, Lambda_mn_t)


#flux
flux_22 = np.diff(Lambda_total_t_22)/np.diff(trange)
flux_11 = np.diff(Lambda_total_t_11)/np.diff(trange)
flux_00 = np.diff(Lambda00_t)/np.diff(trange)
flux_super = np.diff(Lambda_total_t_super)/np.diff(trange)




#Probabilities
Pee_22 = r_total_t_22[0] 
Pgg_22 = r_total_t_22[3]

Pee_11 = r_total_t_11[0] 
Pgg_11 = r_total_t_11[3]

Pee_super = r_total_t_super[0] 
Pgg_super = r_total_t_super[3]


#Plotting
my_dpi = 300
fig, ax = plt.subplots(nrows=3, ncols=1, sharex='col', figsize=(2000/my_dpi, 1250/my_dpi), dpi=my_dpi)
#Info
info_string = f"Gamma={Gamma}, Omega={Omega}, t0={t0}, r_sys={r_sys}, r_field_super={r_field_super} "
#Main plot
ax[2].set_xlabel(r'Time ($ t/\Gamma$)')

two_color='red'
one_color='blue'
super_color='purple'
l_width =1
ax[0].plot(trange, Pee_22, linewidth=l_width, color=two_color, linestyle="dotted",  label=r'$N=2$')
ax[0].plot(trange, Pee_11, linewidth=l_width, color= one_color, linestyle="dashed", label=r'$N=1$')
ax[0].plot(trange, Pee_super, linewidth=l_width, color=super_color, linestyle="dashdot", label=r'Superposition')

ax[2].plot(trange, Lambda_total_t_22, linewidth=l_width, color=two_color, linestyle="dotted", label=r'N=2 integrated flux')
ax[2].plot(trange, Lambda_total_t_11, linewidth=l_width, color= one_color,  linestyle="dashed", label=r'N=1 integrated flux')
#ax[0].plot(trange, Lambda00_t, label=r'N=0 integrated flux')
ax[2].plot(trange, Lambda_total_t_super, color=super_color, linewidth=l_width,linestyle="dashdot", label=r'Superposition integrated flux')

ax[1].plot(trange[:-1], flux_22, linewidth=l_width, color=two_color, linestyle="dotted", label=r'N=2 flux')
ax[1].plot(trange[:-1], flux_11, linewidth=l_width, color= one_color,  linestyle="dashed", label=r'N=1 flux')
#ax[0].plot(trange[:-1], flux_00, label=r'N=0 flux')
ax[1].plot(trange[:-1], flux_super, color=super_color, linewidth=l_width, linestyle="dashdot", label=r'Superposition flux')

ax[0].plot(trange, pulse_func(trange)**2, alpha=1, linewidth=0.5, linestyle="solid", zorder=0, color="black", label=r'$|\xi(t)|^2$')
ax[0].fill_between(trange, 0, pulse_func(trange)**2, color="black", alpha=0.1)
ax[1].plot(trange, pulse_func(trange)**2, alpha=1, linewidth=0.5, linestyle="solid", zorder=0, color="black", label=r'$|\xi(t)|^2$')
ax[1].fill_between(trange, 0, pulse_func(trange)**2, color="black", alpha=0.1)
ax[2].plot(trange, pulse_func(trange)**2, alpha=1, linewidth=0.5, linestyle="solid", zorder=0, color="black", label=r'$|\xi(t)|^2$')
ax[2].fill_between(trange, 0, pulse_func(trange)**2, color="black", alpha=0.1)


ax[0].set_xlim(left=0, right=tmax)
ax[0].set_ylim([0,1])
ax[1].set_ylim([0,1.1])
ax[2].set_ylim([0,2])

from matplotlib.ticker import MaxNLocator

ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))

#yticks[1].label1.set_visible(False)
#ax[0].legend(prop={'size': 8})

#for i in range (0,3):
  #box = ax[i].get_position()
  #ax[i].set_position([box.x0, box.y0 + box.height - 0.1,
                 #box.width, box.height * 0.9])
  #ax[i].set_position([box.x0 - box.width * 0.1, box.y0,
  #               box.width *0.9, box.height])

handles, labels = ax[0].get_legend_handles_labels()
# Put a legend below current axis
#fig.legend(handles, labels, loc='upper center',
          #fancybox=False, shadow=False, ncol=1, prop={'size': 8})
ax[0].legend(handles, labels, frameon=False,  fontsize='xx-small') 
#Info plot
#ax[1].axis('off')
#ax[1].annotate(info_string, xy=(0.5, 0), xytext=(0, 10), xycoords=('axes fraction', 'figure fraction'),
#            textcoords='offset points', ha='center', va='bottom')
#Output
file_num = 0
while os.path.exists("plot%s.png" % file_num):
    file_num += 1
plt.tight_layout()
fig.savefig(str("plot"+str(file_num)))
plt.show()

