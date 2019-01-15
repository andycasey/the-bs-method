


import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from spectrum import Spectrum1D

np.random.seed(0)


wavelength, tolerance = (6141.71, 10)
solar_spectrum = Spectrum1D.read("data/spectra/0_sun_n.fits")


# Take some data around the line.
idx = np.searchsorted(solar_spectrum.dispersion, 
                      [wavelength - tolerance, wavelength + tolerance])

x = solar_spectrum.dispersion[idx[0]:idx[1]]
y = solar_spectrum.flux[idx[0]:idx[1]]
yerr = np.abs(np.random.normal(0, 0.01/3, size=y.size))

P = y.size




# Plot the distributions I imagine.
from scipy import stats

xi = np.linspace(0, 1.1, 1000)
pdf_normal = stats.norm.pdf(xi, 1, 0.01)

# Specifically, powerlaw.pdf(x, a, loc, scale) is identically equivalent to 
#               powerlaw.pdf(y, a) / scale with y = (x - loc) / scale.

fig, ax = plt.subplots()
ax.plot(xi, pdf_normal, c="tab:blue")

for a in [0.5, 1, 2, 5, 10, 50, 100]:
    pdf_powerlaw = stats.powerlaw.pdf(xi, a=a, loc=0.05, scale=1)

    ax.plot(xi, pdf_powerlaw, label=f"a = {a}")

plt.legend()

# OK, so we just want a > 1

# Let's plot the solar spectrum and fit a two component mixture model to that.


fig, ax = plt.subplots()
ax.hist(y, bins=20)

