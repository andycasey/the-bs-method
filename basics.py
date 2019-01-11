
"""
The Bedell-Spina method.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy import stats
from spectrum import Spectrum1D
from scipy import optimize as op


np.random.seed(0)

spectrum = Spectrum1D.read("data/spectra/0_sun_n.fits")

fig, ax = plt.subplots()
ax.hist(1 - spectrum.flux, bins=50)


# load line list
wls = np.loadtxt("data/line-lists/sun_2015_05_18-2.moog.edited", usecols=(0, ))
#raise a



line_wavelength, window, snr = (np.random.choice(wls), 5, 1000)
#line_wavelength = 6186.7
line_wavelength = np.random.choice(wls)

idx = np.searchsorted(spectrum.dispersion, 
                      [line_wavelength - window, line_wavelength + window])

x = spectrum.dispersion[idx[0]:idx[1]]
y = spectrum.flux[idx[0]:idx[1]]
y_err = np.ones_like(x) * 0.001 # np.abs(np.random.normal(0, 2.35/snr, size=y.size))

LINE_WAVELENGTH_TOLERANCE = 0.1



def gaussian(x, x_0, amplitude, sigma, **kwargs):
    return amplitude * np.exp(-(x - x_0)**2 / (2.0 * sigma**2))


def continuum(x, continuum_coefficients):
    if not len(continuum_coefficients):
        return np.ones_like(x)
    return np.polyval(continuum_coefficients, x)


def stellar_flux(x, x_0, amplitude, sigma, continuum_coefficients, **kwargs):
    normalised_flux = 1 - gaussian(x, x_0, amplitude, sigma)
    return continuum(x, continuum_coefficients) * normalised_flux


parameter_names = ("w", "x_0", "amplitude", "sigma",
                    "ln_outlier_sigma", "outlier_mean",
                   "continuum_coefficients")


def pack(*parameters):
    packed = OrderedDict()
    P, N = (len(parameters), len(parameter_names))

    for parameter_name, v in zip(parameter_names, parameters):
        packed[parameter_name] = v

    if P >= N:
        packed[parameter_names[-1]] = parameters[N - 1:]

    packed.setdefault(parameter_names[-1], [])
    packed[parameter_names[-1]] = np.atleast_1d(packed[parameter_names[-1]])

    assert "amplitude" in packed
    return packed


def unpack(parameters):
    return np.hstack([parameters[k] for k in parameter_names])


DEFAULT_BOUNDS = dict(w=[0, 1],
                      x_0=[line_wavelength - LINE_WAVELENGTH_TOLERANCE, line_wavelength + LINE_WAVELENGTH_TOLERANCE], # MAGIC REFACTOR
                      amplitude=[0, 1],
                      amplitude_L=[0, 2],
                      fwhm_G=[0, np.inf],
                      fwhm_L=[0, np.inf],
                      sigma=[0, np.inf],
                      ln_outlier_sigma=[-10, 5])

# objective function
def ln_prior(x, *parameters, bounds=None, full_output=False, **kwargs):

    theta = pack(*parameters)
    
    # Check bounds.
    bounds = bounds or DEFAULT_BOUNDS
    for k, (lower, upper) in bounds.items():
        if k in theta and not (upper >= theta[k] >= lower):
            return -np.inf if not full_output else (-np.inf, k)

    # Check minimum mixing weight.
    #if theta["w"] < (10 * 0.1)/np.ptp(x):
    #    return -np.inf if not full_output else (-np.inf, "w_check")

    # Check outlier_mean minimum.
    if theta["outlier_mean"] < (np.log(5 * np.mean(y_err)) + np.exp(theta["ln_outlier_sigma"])**2):
        return -np.inf

    # Check continuum.
    if np.any(continuum(x, theta["continuum_coefficients"]) < 0):
        return -np.inf if not full_output else (-np.inf, "continuum")

    return 0



def ln_likelihood(x, y, y_err, *parameters, **kwargs):
    
    theta = pack(*parameters)
    print(theta)
    is_line = (np.abs(x - line_wavelength) <= 0.20)

    model = stellar_flux(x, **theta)
    residuals = (model - y)

    ivar = y_err**-2
    ll_foreground = -0.5 * (model - y)**2 * ivar - 0.5 * np.log(2 * np.pi * ivar)

    #outlier_mean = -5
    #outlier_sigma = 2

    outlier_mean = theta["outlier_mean"]
    outlier_sigma = np.exp(theta["ln_outlier_sigma"])

    ll_background = -np.log(residuals * outlier_sigma * np.sqrt(2 * np.pi)) \
                    - 0.5 * ((np.log(residuals) - outlier_mean)/outlier_sigma)**2


    Q = theta["w"]
    lls = np.array([
        np.log(Q) + ll_foreground,
        np.log(1 - Q) + ll_background
    ])
    lls[1, is_line] = np.nanmin(lls)
    lls[~np.isfinite(lls)] = np.nanmin(lls)


    ll = np.sum(np.logaddexp(*lls))
    return ll



def ln_probability(x, y, y_err, *parameters, **kwargs):

    lp = ln_prior(x, *parameters, **kwargs)
    if not np.isfinite(lp): return lp

    r = lp + ln_likelihood(x, y, y_err, *parameters, **kwargs)
    print(parameters, r)
    return r




def get_p0():

    sigma = 0.05
    ln_outlier_sigma = 2

    outlier_mean = 0.5 + (np.log(3 * np.mean(y_err)) + np.exp(ln_outlier_sigma)**2)

    continuum_order = 0

    return unpack(dict(w=0.5,
                       x_0=line_wavelength,
                       amplitude=(1 - y[int(y.size/2)]),
                       amplitude_L=(1 - y[int(y.size/2)]),
                       sigma=sigma,
                       fwhm_G=2.35 * sigma,
                       fwhm_L=2.35 * sigma,
                       outlier_mean=outlier_mean,
                       ln_outlier_sigma=ln_outlier_sigma,
                       continuum_coefficients=np.polyfit(x, y, continuum_order)))




p0 = get_p0()
nlp = lambda p: -ln_probability(x, y, y_err, *p)

p_opt = op.minimize(nlp, p0, method="BFGS")

while True:
    p_opt = op.minimize(nlp, p_opt.x, method="Nelder-Mead")
    if p_opt.success: break

print(pack(*p_opt.x))

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(x, y, c='k')
ax.plot(x, stellar_flux(x, **pack(*p0)), c="tab:blue")
ax.plot(x, stellar_flux(x, **pack(*p_opt.x)), c="tab:red")

y_mod = stellar_flux(x, **pack(*p_opt.x))

fig, ax = plt.subplots()
ax.hist(y_mod - y, bins=50)

outlier_mean = pack(*p_opt.x)["outlier_mean"]
outlier_sigma = np.exp(pack(*p_opt.x)["ln_outlier_sigma"])**0.5

xi = np.percentile(y_mod - y, np.linspace(0, 100, 1000))

pdf = np.exp(-(np.log(xi) - outlier_mean)**2 / (2 * outlier_sigma**2)) \
    / (xi * outlier_sigma * np.sqrt(2 * np.pi))

fig, ax = plt.subplots()
ax.plot(xi, pdf, c="r")
ax.plot(xi, stats.lognorm.pdf(xi, outlier_sigma, scale=np.exp(outlier_mean)),
        c="tab:blue", alpha=0.5)

