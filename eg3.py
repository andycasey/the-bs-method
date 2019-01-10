
import numpy as np
from spectrum import Spectrum1D
import stan_utils

import matplotlib.pyplot as plt

random_seed = 10

np.random.seed(random_seed)

sun = Spectrum1D.read("data/spectra/0_sun_n.fits")

# random line: 5044.211
line_wavelength = 5466.987
line_wavelength = 5044.211
line_wavelength =   6082.711 
line_wavelength =  4602.001
#line_wavelength =  4365.896
line_wavelength = 4602.9
line_wavelength =  6141.71
wavelength_tolerance = 0.05
window = 10
continuum_order = 0

small = 1e-10
snr = 100


# Take some data around the line.
idx = np.searchsorted(sun.dispersion, [line_wavelength - window, line_wavelength + window])

wavelength = sun.dispersion[idx[0]:idx[1]]
flux = sun.flux[idx[0]:idx[1]]
flux_error = np.abs(np.random.randn(flux.size) * (flux/snr))
P = len(flux)

model = stan_utils.load_stan_model(f"{__file__[:-3]}.stan")


data_dict = dict(P=flux.size,
                 wavelength=wavelength,
                 flux=flux,
                 flux_error=flux_error,
                 line_wavelength=line_wavelength,
                 line_wavelength_bounds=(line_wavelength - wavelength_tolerance,
                                         line_wavelength + wavelength_tolerance),
                 continuum_order=continuum_order,
                 small=small,
                 scalar=3)

init_dict = dict(line_wavelength=line_wavelength,
                 continuum_coefficients=np.hstack([1, np.zeros(continuum_order)]),
                 line_amplitude=1 - flux[int(P/2)],
                 line_sigma=0.05,
                 outlier_mean=-2,
                 log_outlier_sigma=-9,
                 theta=0.5)


op_kwds = dict(tol_obj=1e-16,
               tol_rel_grad=1e-16, 
               tol_rel_obj=1e-16, 
               init_alpha=1,
               seed=random_seed, 
               iter=1000000)

opt = model.optimizing(data=data_dict, init=init_dict, **op_kwds)

print_keys = ['continuum_coefficients', 'line_wavelength', 'line_amplitude', 
              'line_sigma', "outlier_mean", 'outlier_sigma', 'theta']

for k in print_keys:
    if k in opt:
        print(f"{k}: {opt[k]}")


fig, ax = plt.subplots()
ax.axvline(line_wavelength, c="#666666", linewidth=0.5, linestyle=":")
ax.axvspan(line_wavelength - wavelength_tolerance,
           line_wavelength + wavelength_tolerance,
           facecolor="#666666", alpha=0.5, zorder=-1)
ax.plot(wavelength, flux, c='k')
ax.plot(wavelength, opt["continuum"], c="tab:blue", alpha=0.5)
ax.plot(wavelength, opt["model_flux"], c="tab:blue")
ax.plot(wavelength, opt["p_outlier"], c="tab:red", alpha=0.5)


