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

Z = np.loadtxt('data.txt')
n, m = Z.shape
l = 1

x = np.linspace(-l/2, l/2, m)
y = np.linspace(-l/2, l/2, n)

X, Y = np.meshgrid(x, y)


phi1, phi1_u, dx, dy = process(x, y, Z, **processing_settings)

Z0 = np.loadtxt('data_reference.txt')

if use_equal_windows:
    phi0, phi0_u, _, _ = process(x, y, Z0, dx=dx, dy=dy, **processing_settings)
else:
    phi0, phi0_u, _, _ = process(x, y, Z0, **processing_settings)

#%%
dphi_u = phi1_u-phi0_u
dphi = phi1-phi0

plt_dphi_u = dphi_u

plt.contourf(x, y, plt_dphi_u)
plt.gca().set_aspect('equal')
plt.colorbar()
plt.title('Unwrapped phase  difference [rad/$\pi$]')

mn = dphi_u.mean()
md = np.ma.median(dphi_u)

phi1_mn = np.ma.median(dphi_u[dphi_u<mn])
phi2_mn = np.ma.median(dphi_u[dphi_u>=mn])
dphi_mn = phi2_mn - phi1_mn

phi1_md = np.ma.median(dphi_u[dphi_u<md])
phi2_md = np.ma.median(dphi_u[dphi_u>=md])
dphi_md = phi2_md - phi1_md

dphi_ptp = dphi_u.ptp()
print(f'Median phase difference (splitted by median): {dphi_md} (2pi modulo: {np.mod(dphi_md, 2*np.pi)})')
print(f'Median phase difference (splitted by mean): {dphi_mn} (2pi modulo: {np.mod(dphi_mn, 2*np.pi)})')
print(f'Peak to peak phase difference: {dphi_ptp} (2pi modulo: {np.mod(dphi_ptp, 2*np.pi)})')
plt.gcf().tight_layout()
plt.show()