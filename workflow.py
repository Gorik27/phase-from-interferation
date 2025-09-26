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
'mask_coef' : 0.75,     # The level of which is determined by peaks (all that is higher than mask_coef*maximum value)
'window_coef' : 10,     # How much the window width is greater than the peak width of the mask_coef level 
'freq_window' : 'hann'  # Choosing a filter window
                        # Options: 'no' - without a window
                        # 'hann' - Hann window
                        # 'bh' - blackmanharris (I dont prefer it)
                        # 'cos' - xosine window
                        # 'tukey' - tukey with parameter equal to 0.5 (flat in the middle)
}

N = 1000
n, m = N, N
lmb = 400*1e-9 #nm

r1_i = 0.3 # Reflection from the film (center)
r2_i = 0.6 # Reflection from the substrate (edge)
drho_i = 1 # The difference in phases of amplitude coefficients of reflection of the substrate and film (layered system)

l = 2*1e-2 #cm - The width of the screen
R1 = 0.5 # radius of the film (center) in shares from L
R2 = R1+0.4 # radius (thickness)

R1*=l/2
R2*=l/2

theta = 1*1e-3 #mrads - angle between interfering rays
psi = np.pi*(1/4+1/8) # The angle between the plum of the fall of light and the axis x

P = lmb/2/theta 
p = l/n

print(f'Period (P): {P*1e2} cm')
print(f'Pixel size (p): {p*1e2} cm')
print(f'p < P/6? ----> {p<P/6}')



r1 = r1_i
r2 = r2_i
k = 2*np.pi/lmb
rho1 = drho_i
rho2 = 0
dz = 0#lmb*0.01


x = np.linspace(-l/2, l/2, n)
y = np.linspace(-l/2, l/2, m)

X, Y = np.meshgrid(x, y)


L = np.sin(theta)*(np.cos(psi)*X + np.sin(psi)*Y)
Z = np.zeros((n, m))
mask1 = (X*X+Y*Y<R1*R1)
mask2 = (~mask1)*(X*X+Y*Y<R2*R2)

Z[mask1] += 1+r1+2*np.sqrt(r1)*np.cos(rho1-k*(L[mask1]+2*dz))
Z[mask2] += 1+r2+2*np.sqrt(r2)*np.cos(rho2-k*(L[mask2]+2*dz))

S1, phi1_u, dx, dy = process(x, y, Z, **processing_settings)

r1 = r1_i
r2 = r1_i
rho1 = 0
rho2 = 0
dz = 0#lmb*0.01

L = np.sin(theta)*(np.cos(psi)*X + np.sin(psi)*Y)
Z = np.zeros((n, m))
mask1 = (X*X+Y*Y<R1*R1)
mask2 = (~mask1)*(X*X+Y*Y<R2*R2)

Z[mask1] += 1+r1+2*np.sqrt(r1)*np.cos(rho1-k*(L[mask1]+2*dz))
Z[mask2] += 1+r2+2*np.sqrt(r2)*np.cos(rho2-k*(L[mask2]+2*dz))

if use_equal_windows:
    S0, phi0_u, _, _ = process(x, y, Z, dx=dx, dy=dy, **processing_settings)
else:
    S0, phi0_u, _, _ = process(x, y, Z, **processing_settings)

#%%
#remove_skikes = True

dphi_u = phi1_u-phi0_u

if remove_spikes:
    va, vb = np.ma.median(dphi_u[mask1]), np.ma.median(dphi_u[mask2])
    v1, v2 = min(va, vb)-1, max(va, vb)+1
    pltmask = (dphi_u < v1) + (dphi_u > v2)
    plt_dphi_u = np.ma.array(dphi_u, mask=pltmask)
else:
    plt_dphi_u = dphi_u


plt.contourf(x, y, plt_dphi_u)
plt.gca().set_aspect('equal')
plt.colorbar()
plt.title('Unwrapped phase  difference [rad/$\pi$]')

dphi_m = np.ma.median(dphi_u[mask1])-np.ma.median(dphi_u[mask2])
dphi_m *= -1
print(f'Median phase difference: {dphi_m} (2pi modulo: {np.mod(dphi_m, 2*np.pi)})')
plt.gcf().tight_layout()
plt.show()