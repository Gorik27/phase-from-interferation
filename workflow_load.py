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


name = 'groove.txt'
name_reference = 'groove_reference.txt'


use_equal_windows = 1 # make a window a second time equal to the first

processing_settings = { #The accuracy of the result greatly depends on the following parameters of processing and discharge +1 peak
'mask_coef' : 0.75,     # The level of which is determined by peaks (all that is higher than mask_coef*maximum value)
'window_coef' : 7.5,     # How much the window width is greater than the peak width of the mask_coef level 
'freq_window' : 'hann'  # Choosing a filter window
                        # Options: 'no' - without a window
                        # 'hann' - Hann window
                        # 'bh' - blackmanharris (I dont prefer it)
                        # 'cos' - xosine window
                        # 'tukey' - tukey with parameter equal to 0.5 (flat in the middle)
}

Z = np.loadtxt(name)
mask = (Z!=0)
noize_scale = 0.0
Z[mask] *= np.random.normal(loc=1, scale=noize_scale, size=Z.shape)[mask]
n, m = Z.shape
l = 1

x = np.linspace(-l/2, l/2, m)
y = np.linspace(-l/2, l/2, n)

X, Y = np.meshgrid(x, y)


S1, phi1_u, dx, dy = process(x, y, Z, **processing_settings)

Z0 = np.loadtxt(name_reference)
mask0 = (Z0!=0)
Z0[mask0] *= np.random.normal(loc=1, scale=noize_scale, size=Z.shape)[mask]

if use_equal_windows:
    S0, phi0_u, _, _ = process(x, y, Z0, dx=dx, dy=dy, **processing_settings)
else:
    S0, phi0_u, _, _ = process(x, y, Z0, **processing_settings)

#%%
S = (S0+S1)/2

dphi_u = phi1_u-phi0_u
dphi_u -= dphi_u.min()

mn = S.mean()
md = np.ma.median(S)
mask_md = (S<md)

xs = X[~mask_md].ravel()
ys = Y[~mask_md].ravel()


phi1_md = np.ma.median(dphi_u[~mask_md])
phi2_md = np.ma.median(dphi_u[mask_md])
dphi_md = phi2_md - phi1_md

remove_spikes = 1
if remove_spikes:
    da = 10
    va, vb = np.ma.median(dphi_u[mask_md]), np.ma.median(dphi_u[~mask_md])
    v1, v2 = min(va, vb)-da, max(va, vb)+da
    pltmask = (dphi_u < v1) + (dphi_u > v2)
    plt_dphi_u = np.ma.array(dphi_u, mask=pltmask)
else:
    plt_dphi_u = dphi_u

#plt_dphi_u = dphi_u

plt.subplot(121)
plt.contourf(x, y, plt_dphi_u)
plt.gca().set_aspect('equal')
plt.colorbar()
plt.title('Unwrapped phase  difference [rad/$\pi$]')

plt.subplot(122)
plt.contourf(x, y, S)
nn = 5
plt.plot(xs[::nn], ys[::nn], '.', alpha=0.005, color='black')
plt.gca().set_aspect('equal')
plt.colorbar()
plt.title('S2')


dphi_ptp = dphi_u.ptp()
print(f'Median phase difference (splitted by median): {dphi_md} (2pi modulo: {np.mod(dphi_md, 2*np.pi)})')
print(f'Peak to peak phase difference: {dphi_ptp} (2pi modulo: {np.mod(dphi_ptp, 2*np.pi)})')
plt.gcf().tight_layout()
plt.show()