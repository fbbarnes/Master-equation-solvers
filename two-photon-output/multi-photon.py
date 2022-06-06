import numpy as np
import scipy as sc
from scipy import integrate
from scipy.stats import poisson
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import uuid
import os
from warnings import warn
from tqdm import tqdm
import datetime
import pickle
from tkinter import filedialog as fd

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

def prop_state_flux(t, y, pbar, state):
  last_t, dt = state
  n = int((t - last_t)/dt)
  pbar.update(n)
  state[0] = last_t +dt * n




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

def create_coh_field(fock_size, mu):
  r_field = np.zeros((fock_size, fock_size))
  for i in range(0, fock_size):
    print(poisson.pmf(i, mu))
    r_field[i][i] =  poisson.pmf(i, mu)

  return r_field

def create_system_state(fock_size, sys_size, r_sys):
  rho_init = np.zeros((fock_size, fock_size, sys_size**2))
  r_sys = 1 * np.outer(psi_g, psi_g).reshape(-1) +  0 * np.outer(psi_e, psi_e).reshape(-1) +  0 * np.outer(psi_e, psi_g).reshape(-1) +  0 * np.outer(psi_g, psi_e).reshape(-1)
  for m in range(0, fock_size):
    for n in range(0, fock_size):
      if m == n:
        rho_init[m][n] = r_sys
      else:
        rho_init[m][n] = np.zeros(sys_size**2)
  
  return rho_init 




#Wavepacket
Omega = 10
t0 = 0
pulse_func = lambda x: gaus(x, Omega, t0)

#Field state(s)
fock_size = 102
#two photon
r_field_22 = np.zeros((fock_size, fock_size))
r_field_22[2][2] = 1
#single photon
r_field_11 = np.zeros((fock_size, fock_size))
r_field_11[1][1] = 1
#superposition of one and two
r_field_super = 0.5 * r_field_11 + 0.5 * r_field_22

#four photon
r_field_44 = np.zeros((fock_size, fock_size))
r_field_44[4][4] = 1
#ten photon
r_field_10 = np.zeros((fock_size, fock_size))
r_field_10[10][10] = 1
#twenty photon
r_field_20 = np.zeros((fock_size, fock_size))
r_field_20[20][20] = 1
#hundred photon
r_field_100 = np.zeros((fock_size, fock_size))
r_field_100[100][100] = 1


#coherent state
r_field_coh10 = create_coh_field(fock_size=fock_size, mu=10)
r_field_coh50 = create_coh_field(fock_size=fock_size, mu=50)
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
rho_init = create_system_state(fock_size=fock_size, sys_size=sys_size, r_sys=r_sys)

initial_fock_dens = np.reshape(rho_init, -1)

#Time settings
tmin = -20
tmax = 40
trange = np.linspace(tmin, tmax, 6000)


#Solver
initial_conditions = np.hstack((initial_fock_dens, initial_flux_expectations))
with tqdm(total=1000, unit="‰") as pbar:
  sol = sc.integrate.solve_ivp(prop_state_flux, [tmin,tmax], initial_conditions, t_eval=trange,max_step=0.05,args=[pbar, [tmin, (tmax-tmin)/1000]])

now = datetime.datetime.now()
time = now.strftime("%Y-%m-%d--%H-%M-%S")
print("Current Time =", time)
filepath =  str(time) + '--multi-photon-sol.obj' 
print(filepath)
file = open(filepath, 'wb')
pickle.dump(sol, file)
file.close()
'''

openfile = fd.askopenfilename()
sol = np.load(openfile, allow_pickle=True)
'''
#Final states
rho_mn_t = sol.y[0:(fock_size**2 * sys_size**2)]
rho_mn_t = np.reshape(rho_mn_t, (fock_size, fock_size, sys_size**2, np.size(trange)))

#Final flux expectations
Lambda_mn_t = sol.y[(fock_size**2 * sys_size**2):]
Lambda_mn_t = np.reshape(Lambda_mn_t, (fock_size, fock_size, np.size(trange)))

#Total states
r_total_t_22 = weight(r_field_22, rho_mn_t)
r_total_t_11 = weight(r_field_11, rho_mn_t)
r_total_t_super = weight(r_field_super, rho_mn_t)

r_total_t_44 = weight(r_field_44, rho_mn_t)
r_total_t_10 = weight(r_field_10, rho_mn_t)
r_total_t_20 = weight(r_field_20, rho_mn_t)
r_total_t_100 = weight(r_field_100, rho_mn_t)
r_total_t_coh10 = weight(r_field_coh10, rho_mn_t)
r_total_t_coh50 = weight(r_field_coh50, rho_mn_t)

#Total fluxes
Lambda_total_t_22 = weight(r_field_22, Lambda_mn_t)
Lambda_total_t_11 = weight(r_field_11, Lambda_mn_t)
Lambda_total_t_super = weight(r_field_super, Lambda_mn_t)

Lambda_total_t_44 = weight(r_field_44, Lambda_mn_t)
Lambda_total_t_10 = weight(r_field_10, Lambda_mn_t)
Lambda_total_t_20 = weight(r_field_20, Lambda_mn_t)
Lambda_total_t_100 = weight(r_field_100, Lambda_mn_t)
Lambda_total_t_coh10 = weight(r_field_coh10, Lambda_mn_t)
Lambda_total_t_coh50 = weight(r_field_coh50, Lambda_mn_t)



#flux
flux_22 = np.diff(Lambda_total_t_22)/np.diff(trange)
flux_11 = np.diff(Lambda_total_t_11)/np.diff(trange)
flux_super = np.diff(Lambda_total_t_super)/np.diff(trange)

flux_44 = np.diff(Lambda_total_t_44)/np.diff(trange)
flux_10 = np.diff(Lambda_total_t_10)/np.diff(trange)
flux_20 = np.diff(Lambda_total_t_20)/np.diff(trange)
flux_100 = np.diff(Lambda_total_t_100)/np.diff(trange)
flux_coh10 = np.diff(Lambda_total_t_coh10)/np.diff(trange)
flux_coh50 = np.diff(Lambda_total_t_coh50)/np.diff(trange)


#Probabilities
Pee_22 = r_total_t_22[0] 
Pgg_22 = r_total_t_22[3]

Pee_11 = r_total_t_11[0] 
Pgg_11 = r_total_t_11[3]

Pee_super = r_total_t_super[0] 
Pgg_super = r_total_t_super[3]


Pee_44 = r_total_t_44[0] 
Pgg_44 = r_total_t_44[3]

Pee_10 = r_total_t_10[0] 
Pgg_10 = r_total_t_10[3]

Pee_20 = r_total_t_20[0] 
Pgg_20 = r_total_t_20[3]

Pee_100 = r_total_t_100[0] 
Pgg_100 = r_total_t_100[3]

Pee_coh10 = r_total_t_coh10[0] 
Pgg_coh10 = r_total_t_coh10[3]

Pee_coh50 = r_total_t_coh50[0] 
Pgg_coh50 = r_total_t_coh50[3]


#Plotting
my_dpi = 300
fig, ax = plt.subplots(nrows=3, ncols=1, sharex='col', figsize=(2000/my_dpi, 1250/my_dpi), dpi=my_dpi)
#Info
info_string = f"Gamma={Gamma}, Omega={Omega}, t0={t0}, r_sys={r_sys}, r_field_super={r_field_super} "
#Main plot
ax[2].set_xlabel(r'Time ($ t/\Gamma$)')

two_color='red'
one_color='blue'
ten_color='violet'
twenty_color='purple'
hun_color='red'
coh10_colour='gold'
coh50_colour='orange'
four_color='black'
l_width =1
#ax[0].plot(trange, Pee_22, linewidth=l_width, color=two_color, linestyle="dotted",  label=r'$N=2$')
ax[0].plot(trange, Pee_11, linewidth=l_width, color= one_color, linestyle="dashed", label=r'$N=1$')
#ax[0].plot(trange, Pee_super, linewidth=l_width, color=super_color, linestyle="dashdot", label=r'Superposition')

ax[0].plot(sol.t, Pee_10, linewidth=l_width, color=ten_color, linestyle="dashdot", label=r'$N=10$')
ax[0].plot(sol.t, Pee_20, linewidth=l_width, color=twenty_color, linestyle="dashdot", label=r'$N=20$')
ax[0].plot(sol.t, Pee_100, linewidth=l_width, color=hun_color, linestyle="dashdot", label=r'$N=100$')
ax[0].plot(sol.t, Pee_coh10, linewidth=l_width, color=coh10_colour, linestyle="dashdot", label=r'Coh 10')
ax[0].plot(sol.t, Pee_coh50, linewidth=l_width, color=coh50_colour, linestyle="dashdot", label=r'$Coh 50')



ax[2].plot(sol.t, Lambda_total_t_22, linewidth=l_width, color=two_color, linestyle="dotted", label=r'N=2 integrated flux')
ax[2].plot(sol.t, Lambda_total_t_11, linewidth=l_width, color= one_color,  linestyle="dashed", label=r'N=1 integrated flux')
'''
ax[2].plot(trange, Lambda_total_t_44, color=four_color, linewidth=l_width,linestyle="dashdot", label=r'N=4 integrated flux')
'''

ax[1].plot(sol.t[:-1], flux_22, linewidth=l_width, color=two_color, linestyle="dotted", label=r'N=2 flux')
ax[1].plot(sol.t[:-1], flux_11, linewidth=l_width, color= one_color,  linestyle="dashed", label=r'N=1 flux')
'''
ax[1].plot(trange[:-1], flux_44, color=four_color, linewidth=l_width, linestyle="dashdot", label=r'N=4 flux')
'''

ax[0].plot(sol.t, pulse_func(trange)**2, alpha=1, linewidth=0.5, linestyle="solid", zorder=0, color="black", label=r'$|\xi(t)|^2$')
ax[0].fill_between(sol.t, 0, pulse_func(trange)**2, color="black", alpha=0.1)
ax[1].plot(sol.t, pulse_func(trange)**2, alpha=1, linewidth=0.5, linestyle="solid", zorder=0, color="black", label=r'$|\xi(t)|^2$')
ax[1].fill_between(sol.t, 0, pulse_func(trange)**2, color="black", alpha=0.1)
ax[2].plot(sol.t, pulse_func(trange)**2, alpha=1, linewidth=0.5, linestyle="solid", zorder=0, color="black", label=r'$|\xi(t)|^2$')
ax[2].fill_between(sol.t, 0, pulse_func(trange)**2, color="black", alpha=0.1)


ax[0].set_xlim(left=-20, right=40)
ax[0].set_ylim([0,1])
ax[1].set_ylim([0,fock_size])
ax[2].set_ylim([0,fock_size])

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