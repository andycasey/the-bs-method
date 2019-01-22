

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from collections import OrderedDict
from inspect import getfullargspec
from scipy import (optimize as op, special, stats)

from matplotlib.ticker import MaxNLocator

from mpl_utils import mpl_style



from spectrum import Spectrum1D
wavelength, window = (5373.709, 5)

solar_spectrum = Spectrum1D.read("data/spectra/0_sun_n.fits")

# Take some data around the line.
idx = np.searchsorted(solar_spectrum.dispersion, 
                      [wavelength - window, wavelength + window])

x = solar_spectrum.dispersion[idx[0]:idx[1]]
y_c = solar_spectrum.flux[idx[0]:idx[1]]
y_c_err = np.abs(np.random.normal(0, 1e-3, size=y_c.size))


coeffs = np.array([  1.25739993e+01,  -1.35046407e+05,   3.62610095e+08])
true_continuum = np.polyval(coeffs, x)

y = true_continuum * y_c
y_err = true_continuum**2 * y_c_err

coeffs = np.polyfit(x, y, 2)
continuum = np.polyval(coeffs, x)
#f = continuum * y_c


figsize = (7.13, 2)

fig, ax = plt.subplots(figsize=figsize)

ax.plot(x, y, "-", c="k", drawstyle="steps-mid")
ax.plot(x, true_continuum, "-", c="tab:blue")
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(5))

ax.set_xlim(x[0], x[-1])
ylim = ax.get_ylim()
ax.plot([wavelength, wavelength], ylim, "-", c="#666666", lw=1, linestyle=":", zorder=-1)
ax.set_ylim(ylim)


ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

fig.tight_layout()

fig.savefig("tex/fig1.pdf", dpi=300)



fig, axes = plt.subplots(1, 3, figsize=figsize)

axes[0].hist(y, bins=50, facecolor="#000000", normed=True)
axes[0].set_xlabel(r"$y$")
axes[0].xaxis.set_major_locator(MaxNLocator(5))
axes[0].set_yticks([])
axes[0].set_ylabel(r"$\textrm{frequency}$")


#ys = np.linspace(0, 1.1, y.size) #*  continuum

sigma_g = 2

pdf_f = stats.norm.pdf(y, continuum, y_err)
pdf_f /= np.sum(pdf_f)
pdf_g = stats.lognorm.pdf(1-y/continuum, sigma_g)#, scale=10000)
pdf_g /= np.sum(pdf_g)



idx = np.argsort(y)
axes[1].plot(y[idx], pdf_g[idx], "-", c="tab:red")
axes[1].plot(y[idx], pdf_f[idx], "-", c="tab:blue")


I = 1000
sigma_g = 1.5
yi = np.linspace(0, 1.1, I)

pdf_f = stats.norm.pdf(yi, 1, np.exp(-3))
pdf_f /= np.sum(pdf_f)

pdf_g = stats.lognorm.pdf(1 - yi, sigma_g)
pdf_g /= np.sum(pdf_g)

axes[2].plot(np.ptp(y) * yi + y.min(), pdf_g, "-", c="tab:red")
axes[2].plot(np.ptp(y) * yi + y.min(), pdf_f, "-", c="tab:blue")

