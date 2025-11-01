"""
An example of using the "Process" function for signal processing.

The file generates the interference model signal, and then processes it to
measure the phase difference between area covered with film and area uncovered
"""
from processing import process
import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import matplotlib.pyplot as plt


remove_spikes = 1 #turn on stupid algorithm to remove spikes from final picture (sometimes may not work because it is stupid)
use_equal_windows = 1 # make a window a second time equal to the first

processing_settings = { #The accuracy of the result greatly depends on the following parameters of processing and discharge +1 peak
'scale' : 1000,
'language': 'ru',
'mask_coef' : 0.75,     # The level of which is determined by peaks (all that is higher than mask_coef*maximum value)
'window_coef' : 10,     # How much the window width is greater than the peak width of the mask_coef level 
'freq_window' : 'hann'  # Choosing a filter window
                        # Options: 'no' - without a window
                        # 'hann' - Hann window
                        # 'bh' - blackmanharris (I dont prefer it)
                        # 'cos' - xosine window
                        # 'tukey' - tukey with parameter equal to 0.5 (flat in the middle)
}

def layer_matrix(n, l, lmb):
    phi = 2*np.pi*n*l/lmb
    M = np.array([[np.cos(phi)     , 1j*np.sin(phi)/n],
                  [1j*n*np.sin(phi), np.cos(phi)     ]])
    return M

def reflction(n, l, lmb, nv, ns):
    M = layer_matrix(n, l, lmb)
    m11 = M[0, 0]
    m12 = M[0, 1]
    m21 = M[1, 0]
    m22 = M[1, 1]
    r = (nv*m11 + 1j*nv*ns*m12 - 1j*m21-ns*m22)/(nv*m11 + 1j*nv*ns*m12 + 1j*m21 + ns*m22)
    return r

def r_rho(n, l, lmb, nv, ns):
    R = reflction(n, l, lmb, nv, ns)
    r = np.abs(R)
    rho = np.angle(R)
    return r, rho
    
"""
Material parameters
"""
lmb = 670*1e-9 #nm

#B
n_film = 3.25
k_film = 0.3
N_film = n_film - 1j*k_film

#thickness = 100e-9 #nm # TODO: from 30 to 180 nm
thickness = np.linspace(30, 180)*1e-9

n_air = 1

#Mo
lmb1 = 620e-9
n1 = 0.78
k1 = 8.58
lmb2 = 708e09
n2 = 1.11
k2 = 10.0
n_sub = (n1*(lmb2-lmb) + n2*(lmb-lmb1))/(lmb2-lmb1) # linear interpolation
k_sub = (k1*(lmb2-lmb) + k2*(lmb-lmb1))/(lmb2-lmb1) # linear interpolation
N_sub = n_sub - 1j*k_sub

rho = np.zeros(len(thickness))
for i, t in enumerate(thickness):
    _, rho[i] = r_rho(N_film, t, lmb, n_air, N_sub)
    
plt.figure(dpi=500)
plt.plot(thickness*1e9, rho)
plt.xlabel('B film thickness [nm]')
plt.ylabel('phase')
plt.plot()
#%%
"""
#Signal generation 1
"""
scale = 1000 #mm

N = 1000
n, m = N, N


R1 = 7.5e-3 # radius of the reference mirror (center) [mm]
R2 = 25e-3 # radius of whole mirror [mm]

l = R2*2 + 10e-3 #mm - The width of the screen

x = np.linspace(-l/2, l/2, n)
y = np.linspace(-l/2, l/2, m)
X, Y = np.meshgrid(x, y)
mask1 = (X*X+Y*Y<R1*R1)
mask2 = (~mask1)*(X*X+Y*Y<R2*R2)

#film_modulation = (1+0.2*np.cos(4*np.pi*X/l))[mask2]


theta = 1*1e-3 #mrads - angle between interfering rays
psi = np.pi*(1/4+1/8) # The angle between the plum of the fall of light and the axis x

P = lmb/2/theta 
p = l/n

print(f'Period (P): {P*1e2} cm')
print(f'Pixel size (p): {p*1e2} cm')
print(f'p < P/6? ----> {p<P/6}')

k = 2*np.pi/lmb
dz = 0#lmb*0.1

L = np.sin(theta)*(np.cos(psi)*X + np.sin(psi)*Y)
Z = np.zeros((n, m))

r1, rho1 = r_rho(N_film, 0, lmb, n_air, N_sub)
r2, rho2 = r_rho(N_film, thickness, lmb, n_air, N_sub)


Z[mask1] += 1+r1+2*np.sqrt(r1)*np.cos(rho1-k*(L[mask1]+2*dz))
Z[mask2] += 1+r2+2*np.sqrt(r2)*np.cos(rho2-k*(L[mask2]+2*dz))

"""
#Processing 1
"""

S1, phi1_u, dx, dy = process(x, y, Z, **processing_settings)

"""
#Signal generation 2
"""
r1, rho1 = r_rho(N_film, 0, lmb, n_air, N_sub)
r2, rho2 = r_rho(N_film, 0, lmb, n_air, N_sub)
dz = 0#lmb*0.01

L = np.sin(theta)*(np.cos(psi)*X + np.sin(psi)*Y)
Z = np.zeros((n, m))
mask1 = (X*X+Y*Y<R1*R1)
mask2 = (~mask1)*(X*X+Y*Y<R2*R2)

Z[mask1] += 1+r1+2*np.sqrt(r1)*np.cos(rho1-k*(L[mask1]+2*dz))
Z[mask2] += 1+r2+2*np.sqrt(r2)*np.cos(rho2-k*(L[mask2]+2*dz))

"""
#Processing 2
"""

if use_equal_windows:
    S0, phi0_u, _, _ = process(x, y, Z, dx=dx, dy=dy, **processing_settings)
else:
    S0, phi0_u, _, _ = process(x, y, Z, **processing_settings)

#%%
"""
#Plotting
"""
#remove_skikes = True

dphi_u = phi1_u-phi0_u

if remove_spikes:
    va, vb = np.ma.median(dphi_u[mask1]), np.ma.median(dphi_u[mask2])
    v1, v2 = min(va, vb)-1, max(va, vb)+1
    pltmask = (dphi_u < v1) + (dphi_u > v2)
    plt_dphi_u = np.ma.array(dphi_u, mask=pltmask)
else:
    plt_dphi_u = dphi_u

units = 'a.u.'
match scale:
    case 1000:
        units = 'мм'
    case 100:
        units = 'см'
    case 1:
        units = 'м'
plt.contourf(x, y, plt_dphi_u)
plt.xlabel(f'[{units}]')
plt.ylabel(f'[{units}]')
plt.gca().set_aspect('equal')
plt.colorbar()
plt.title('Развернутая разность фаз [рад/$\pi$]')

dphi_m = np.ma.median(dphi_u[mask1])-np.ma.median(dphi_u[mask2])
dphi_m *= -1
print(f'Median phase difference: {dphi_m} (2pi modulo: {np.mod(dphi_m, 2*np.pi)})')
plt.gcf().tight_layout()
plt.show()