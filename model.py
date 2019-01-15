

import numpy as np
import matplotlib.pyplot as plt
from scipy import (optimize as op, stats)
from collections import OrderedDict
from spectrum import Spectrum1D

np.random.seed(0)


wavelength, window = (6141.71, 10)
line_wavelength_tolerance = 0.10 # MAGIC HACK

solar_spectrum = Spectrum1D.read("data/spectra/0_sun_n.fits")


# Take some data around the line.
idx = np.searchsorted(solar_spectrum.dispersion, 
                      [wavelength - window, wavelength + window])

x = solar_spectrum.dispersion[idx[0]:idx[1]]
y = solar_spectrum.flux[idx[0]:idx[1]]
y_err = np.abs(np.random.normal(0, 0.01, size=y.size))

P = y.size

MAGIC_SCALAR = 0
USE_GAUSSIAN = True
REQUIRE_FG_NEAR_LINE = True

if USE_GAUSSIAN:
    parameter_names = ("Q", 
                       "x0", "amplitude", "sigma", 
                       "ln_intrinsic_scatter",
                       "powerlaw_shape",
                       "continuum_coefficients")

else:

    parameter_names = ("Q", 
                       #"x0", "amplitude", "sigma", 
                       "x0", "amplitude", "fwhm_L", "fwhm_G",
                       "ln_intrinsic_scatter",
                       "powerlaw_shape",
                       "continuum_coefficients")

default_bounds = dict(Q=[0, 1],
                      x0=[
                            wavelength - line_wavelength_tolerance,
                            wavelength + line_wavelength_tolerance,
                         ],
                      amplitude=[0, 1],
                      ln_intrinsic_scatter=[-15, -3],
                      sigma=[0, np.inf],
                      fwhm_G=[0, np.inf],
                      fwhm_L=[0, np.inf],
                      #powerlaw_scale=[0, 0.75],
                      powerlaw_shape=[5, np.inf])


def pack(*parameters):
    packed = OrderedDict()
    for name, value in zip(parameter_names, parameters):
        packed[name] = value
    
    last_parameter = parameter_names[-1]
    P, N = (len(parameters), len(parameter_names))
    if P >= N:
        # Pack remaining into the last parameter name
        packed[last_parameter] = parameters[N-1:]

    packed.setdefault(last_parameter, [])
    packed[last_parameter] = np.atleast_1d(packed[last_parameter])

    return packed


def unpack(parameters):
    return np.hstack([parameters[k] for k in parameter_names])


def gaussian(x, x0, amplitude, sigma, **kwargs):
    return amplitude * np.exp(-(x - x0)**2 / (2.0 * sigma**2))


# model function
def voight(x, x0, amplitude=1, fwhm_L=2/np.pi, fwhm_G=np.log(2), **kwargs):

    A, B, C, D = np.array([
        [-1.2150, -1.3509, -1.2150, -1.3509],  
        [1.2359,   0.3786, -1.2359, -0.3786],    
        [-0.3085,  0.5906, -0.3085,  0.5906],    
        [0.0210,  -1.1858, -0.0210,  1.1858]])

    sqrt_ln2 = np.sqrt(np.log(2))
    X = (x - x0) * 2 * sqrt_ln2 / fwhm_G
    X = np.atleast_1d(X)[..., np.newaxis]
    Y = fwhm_L * sqrt_ln2 / fwhm_G
    Y = np.atleast_1d(Y)[..., np.newaxis]

    V = np.sum((C * (Y - A) + D * (X - B))/(((Y - A) ** 2 + (X - B) ** 2)), -1)

    return (fwhm_L * amplitude * np.sqrt(np.pi) * sqrt_ln2 / fwhm_G) * V


def _continuum(x, continuum_coefficients, **kwargs):
    if not len(continuum_coefficients):
        return np.ones_like(x)
    return np.polyval(continuum_coefficients, x)


def model_stellar_spectrum(x, parameters):

    theta = pack(*parameters)

    if USE_GAUSSIAN:
        normalized_flux = 1 - gaussian(x, **theta)
    
    else:
        normalized_flux = 1 - voight(x, **theta)
    continuum = _continuum(x, **theta)

    return (continuum, normalized_flux)


def ln_prior(x, parameters, bounds=None, full_output=False, **kwargs):

    theta = pack(*parameters)
    bounds = bounds or default_bounds
    for k, (lower, upper) in bounds.items():
        if k in theta and not (upper >= theta[k] >= lower):
            return -np.inf if not full_output \
                           else (-np.inf, k, theta[k], lower, upper)

    # Check continuum.
    continuum = _continuum(x, **theta)
    if np.any(continuum < 0):
        return -np.inf if not full_output \
                       else (-np.inf, "continuum", continuum, 0, np.inf)

    # Check minimum Q:
    if "sigma" in parameter_names:
        sigma = theta["sigma"]
    else:
        sigma = 2.35 * theta["fwhm_G"]
    Q_min = (10 * sigma)/np.ptp(x)
    if theta["Q"] < Q_min:
        return -np.inf if not full_output \
                       else (-np.inf, "Q", theta["Q"], Q_min, 1)


    return 0


def ln_likelihood(x, y, y_err, parameters, full_output=False, **kwargs):

    # TODO: minimize number of packing/unpacking operations
    theta = pack(*parameters)

    continuum, normalized_flux = model_stellar_spectrum(x, parameters)

    foreground = continuum * normalized_flux

    #ivar = 1.0/y_err**2
    #foreground_ll = - 0.5 * (y - foreground)**2 * ivar \
    #                - 0.5 * np.log(2 * np.pi * ivar)
    
    s = np.sqrt(y_err**2 + np.exp(2 * theta["ln_intrinsic_scatter"]))
    foreground_ll = stats.norm.logpdf(y, foreground, s)
    background_ll = stats.powerlaw.logpdf(y/foreground, 
                                          theta["powerlaw_shape"],
                                          loc=MAGIC_SCALAR * s)

    # Require that the line be OK.
    if REQUIRE_FG_NEAR_LINE:
        min_ll = np.nanmin(np.hstack([foreground_ll, background_ll]))
        in_line = np.abs(x - wavelength) <= (5 * theta.get("sigma", theta.get("fwhm_G")))
        background_ll[in_line] = min_ll


    Q = theta["Q"]
    args = [
        np.log(Q) + foreground_ll,
        np.log(1.0 - Q) + background_ll
    ]
    ll = np.sum(np.logaddexp(*args))
    return ll if not full_output else (ll, args, continuum, normalized_flux)


def ln_probability(x, y, y_err, parameters, **kwargs):

    lp = ln_prior(x, parameters, **kwargs)
    if not np.isfinite(lp): return lp

    lp = lp + ln_likelihood(x, y, y_err, parameters, **kwargs)

    print(parameters, lp, pack(*parameters))
    return lp


def get_initial_guess(continuum_order=-1):

    if continuum_order >= 0:
        continuum_coefficients = np.polyfit(x, y, continuum_order)
    else:
        continuum_coefficients = []

    continuum_coefficients = np.hstack([1, np.zeros(continuum_order)])[::-1]
    return unpack(dict(Q=0.9,
                       x0=wavelength,
                       amplitude=1 - y[int(P/2)],
                       sigma=0.10,
                       ln_intrinsic_scatter=-5,
                       powerlaw_shape=10,
                       fwhm_L=0.10,
                       fwhm_G=0.10,
                       continuum_coefficients=continuum_coefficients))


p0 = get_initial_guess(1)
nlp = lambda p: -ln_probability(x, y, y_err, p)

p_opt = op.minimize(nlp, p0,
                    method="Nelder-Mead",
                    options=dict(maxiter=10000,
                                 maxfev=10000,
                                 adaptive=True))

assert p_opt.success

p_opt = op.minimize(nlp, p_opt.x, method="Powell")
assert p_opt.success

theta = pack(*p_opt.x)

continuum, normalized_flux = model_stellar_spectrum(x, p_opt.x)

limits = (0, 1.25)
xi = np.linspace(*limits, 1000)
fig, axes = plt.subplots(2)
s = np.sqrt(y_err**2 + np.exp(2 * theta["ln_intrinsic_scatter"]))
axes[0].plot(xi, stats.powerlaw.pdf(xi, theta["powerlaw_shape"],
                                    loc=MAGIC_SCALAR * np.mean(s)), c="tab:red")
axes[0].plot(xi, stats.powerlaw.pdf(xi, 100), c="tab:blue")
axes[1].hist(y/continuum)

for ax in axes:
    ax.set_xlim(*limits)

fig, ax = plt.subplots()
ax.plot(x, y, c="k")


ax.plot(x, continuum * normalized_flux, c="r")
ax.axvline(wavelength, c="#666666", linestyle=":", linewidth=0.5, zorder=-1)


fig, axes = plt.subplots(3, sharex=True)
ll, (ll_foreground, ll_background), cont, norm_flux = ln_likelihood(x, y, y_err, p_opt.x, full_output=True)

axes[0].plot(x, y)
axes[0].plot(x, cont * norm_flux, c="r")
axes[1].plot(x, ll_foreground)
axes[1].set_ylim(-1000, 1)
axes[2].plot(x, ll_background)
#ax.plot(x, ll_background)

p = np.exp(ll_foreground - np.logaddexp(ll_foreground, ll_background))



fig, ax = plt.subplots()
ax.plot(x, p, c="tab:red", alpha=0.5)
ax.plot(x, y, c="k")
ax.plot(x, cont * norm_flux, c="tab:blue")

c, n = model_stellar_spectrum(x, p0)
#ax.plot(x, c*n, c="tab:green")
raise a





