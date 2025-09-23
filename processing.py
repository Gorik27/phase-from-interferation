"""
This file contains the "Process" function, which takes a signal with the 
interference to the input, and returns a phase map, a deployed phase, a window 
width containing +1 peaks in X and Y (the latter are needed if you need to make
several measurements with the same window width)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2 as fft
from scipy.fft import ifft2 as ifft
from scipy.fft import fftshift
from sklearn.cluster import KMeans
from skimage.restoration import unwrap_phase
from scipy.signal import windows



# The accuracy of the result greatly depends on the following parameters of processing and discharge +1 peak
def process(x, y, Z, 
            dx=False,
            dy=False,
            N_clusters=3,   # The number of peaks in Fourier (normal 3)
            mask_coef=0.75, # The level of which is determined by peaks (all that is higher than mask_coef*maximum value)
            window_coef=10, # How much the window width is greater than the peak width of the mask_coef level 
            freq_window='hann'): # Choosing a filter window
                                 # Options: 'no' - without a window
                                 # 'hann' - Hann window
                                 # 'bh' - blackmanharris (I dont prefer it)
                                 # 'cos' - xosine window
                                 # 'tukey' - tukey with parameter equal to 0.5 (flat in the middle)
    window_coef_x = window_coef
    window_coef_y = window_coef
    X, Y = np.meshgrid(x, y)
    m = len(x)
    n = len(y)
    lx = np.ptp(x)
    ly = np.ptp(y)
    """
    Plot signal
    """
    plt.figure(dpi=500)
    plt.subplot(321)
    plt.gca().set_aspect('equal')
    plt.contourf(x, y, Z)
    plt.colorbar()
    plt.title('Signal')
    mask2 = (Z!=0)
    
    """
    Fourier
    """
    
    F = fftshift(fft(Z))
    Fz = np.log(np.abs(F))
    msk = (Fz>mask_coef*Fz.max())
    plt.subplot(322)
    plt.gca().set_aspect('equal')
    plt.contourf(x, y, Fz)
    plt.title('Fourier')
    plt.colorbar()
    
    """
    Find window consisting +1 peak
    """
    XY = np.column_stack((X[msk].ravel(), Y[msk].ravel()))
    clusters = KMeans(n_clusters=N_clusters, n_init='auto').fit_predict(XY)
    
    cluster_list = list(set(clusters))
    xms, yms = [], []
    for c in cluster_list:
        xy = XY[clusters==c]
        xc, yc = xy[:, 0], xy[:, 1]
        xms.append(xc.mean())
        yms.append(yc.mean())
    
    ci1 = np.argmax(xms)
    ci2 = np.argmax(yms)
    if xms[ci1]/lx > yms[ci2]/ly:
        ci0 = ci1
    else:
        ci0 = ci2
    
    c0 = clusters[ci0]
    
    plt.subplot(323)
    plt.gca().set_aspect('equal')
    plt.scatter(XY[:, 0], XY[:, 1], c=clusters, s=1)
    plt.title('Selecting +1 peak')

    mask = (clusters==c0)
    xs = XY[:, 0][mask]
    ys = XY[:, 1][mask]
    
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    
    #xm, ym = xs.mean(), ys.mean()
    im = np.argmax((Fz[msk]).ravel()[mask])
    xm, ym = xs[im], ys[im]
    
    if dx:
        pass
    else:
        dx = (x2-x1)*window_coef_x
    x1 = xm-dx/2
    x2 = xm+dx/2
    
    if dy:
        pass
    else:
        dy = (y2-y1)*window_coef_y
    y1 = ym-dy/2
    y2 = ym+dy/2
    
    # x1 = xm-(xm-x1)*window_coef_x
    # y1 = ym-(ym-y1)*window_coef_y
    # x2 = xm+(x2-xm)*window_coef_x
    # y2 = ym+(y2-ym)*window_coef_y
    
    plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color='black', alpha=0.3)
    plt.xticks([])
    plt.yticks([])
    plt.plot([xm], [ym], 'x', color='red')
    
    """
    Inverse Fourier and phase calculation
    """
    
    Fzp = np.zeros((n, m))
    Fp = np.zeros((n, m), dtype=np.complex128)
    j1 = np.where(x>x1)[0][0]
    j2 = np.where(x<x2)[0][-1]+1
    i1 = np.where(y>y1)[0][0]
    i2 = np.where(y<y2)[0][-1]+1
    
    dn = (i2-i1)
    n0 = (n-dn)//2
    dm = (j2-j1)
    m0 = (m-dm)//2
    
    
    def get_window(window_type, dn, dm, alpha=False):
        if alpha:
            wind_x = window_type(dm, alpha)
            wind_y = window_type(dn, alpha)
        else:
            wind_x = window_type(dm)
            wind_y = window_type(dn)
        wind_2d = np.outer(wind_y, wind_x)
        return wind_2d
    
    """
    TODO: реализовать круглое окно w_2d(x, y) = w_1d(sqrt(x^2 + y^2))
    """
    # def get_window(window_type, dn, dm, alpha=False):
    #     if alpha:
    #         wind_x = window_type(dm, alpha)
    #         wind_y = window_type(dn, alpha)
    #     else:
    #         wind_x = window_type(dm)
    #         wind_y = window_type(dn)
    #     wind_2d = np.outer(wind_y, wind_x)
    #     return wind_2d
        
        
    
    match freq_window:
        case 'hann':
            print('Hann window is used')
            window = get_window(windows.hann, dn, dm)
        case 'bh':
            print('Blackman-Harris window is used')
            window = get_window(windows.blackmanharris, dn, dm)
        case 'cos':
            print('Cosine window is used')
            window = get_window(windows.cosine, dn, dm)
        case 'tukey':
            print('Tukey window is used')
            alpha = 0.5
            window = get_window(windows.tukey, dn, dm, alpha)
        case 'no':
            window = np.ones((dn, dm))
        case _:
            print('Unkonwn window!!!\nNo window will be used!')
            window = np.ones((dn, dm))
    # plt.figure()
    # plt.contourf(window)
    # plt.show()
    
    
    Fzp[n0:n0+dn, m0:m0+dm] = Fz[i1:i2, j1:j2]*window
    Fp[n0:n0+dn, m0:m0+dm] = F[i1:i2, j1:j2]*window
    
    
    plt.subplot(324)
    plt.gca().set_aspect('equal')
    plt.contourf(x[j1:j2], y[i1:i2], Fz[i1:i2, j1:j2]*window)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('Centered and windowed +1 peak')
    
    Sp = ifft(Fp)
    phi = np.arctan2(np.imag(Sp), np.real(Sp))
    mask = np.ones_like(phi, dtype=bool)
    mask[mask2] = False
    phi = np.ma.array(phi, mask=mask)
    
    plt.subplot(325)
    plt.gca().set_aspect('equal')
    plt.contourf(x, y, phi/(np.pi))
    plt.title('Phase [rad/$\pi$]')
    plt.colorbar()
    
    
    phi_u = unwrap_phase(phi)
    
    plt.subplot(326)
    #np.mod(phi_u, 2*np.pi)
    plt.contourf(x, y, phi_u/(np.pi))
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.title('Unwrapped phase [rad/$\pi$]')

    
    plt.gcf().tight_layout()
    plt.show()
    return phi, phi_u, dx, dy