"""
Time-series statistics for Markov chain Monte Carlo output.

All functions take samples along axis 0; any trailing axes (replicas,
vector components, ...) are treated as independent series.

Conventions
-----------
The integrated autocorrelation time is

    tau_int = 1/2 + sum_{t>=1} rho(t),

so that the variance of the mean of N correlated samples is
``2 * tau_int * var(x) / N``. Uncorrelated samples have ``tau_int = 1/2``.
"""

import numpy as np


def autocorrelation(x):
    """
    Normalized autocorrelation function rho(t) along axis 0 (FFT based).

    Parameters
    ----------
    x : array_like
        Samples, shape ``(n, ...)``.

    Returns
    -------
    rho : ndarray
        Same shape as ``x``; ``rho[0] == 1`` for non-constant series and
        ``rho == 0`` for constant ones.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    dx = x - x.mean(axis=0)
    n_fft = 1 << int(2 * n - 1).bit_length()
    f = np.fft.rfft(dx, n=n_fft, axis=0)
    acov = np.fft.irfft(f * np.conj(f), n=n_fft, axis=0)[:n]
    acov /= n
    var = acov[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(var > 0, acov / np.where(var > 0, var, 1.0), 0.0)
    return rho


def integrated_autocorrelation_time(x, c=6.0):
    """
    Integrated autocorrelation time with Sokal's automatic windowing.

    The window W is the smallest lag with ``W >= c * tau_int(W)``.

    Parameters
    ----------
    x : array_like
        Samples, shape ``(n, ...)``.
    c : float
        Window factor; 5-10 is typical.

    Returns
    -------
    tau : ndarray
        Integrated autocorrelation time, shape ``x.shape[1:]``. Constant
        series give 0.5.
    window : ndarray of int
        Selected window per series.
    """
    rho = autocorrelation(x)
    n = rho.shape[0]
    if n < 3:
        shape = rho.shape[1:]
        return np.full(shape, 0.5), np.ones(shape, dtype=int)

    tau_w = 0.5 + np.cumsum(rho[1:], axis=0)
    lags = np.arange(1, n).reshape((-1,) + (1,) * (rho.ndim - 1))

    ok = lags >= c * tau_w
    has = ok.any(axis=0)
    first = np.where(has, ok.argmax(axis=0), n - 2)
    tau = np.take_along_axis(tau_w, first[None], axis=0)[0]
    tau = np.maximum(tau, 0.5)
    return tau, first + 1


def mean_error(x, tau=None):
    """
    Mean and its standard error, corrected for autocorrelation.

    Parameters
    ----------
    x : array_like
        Samples, shape ``(n, ...)``.
    tau : array_like, optional
        Integrated autocorrelation time; estimated if omitted.

    Returns
    -------
    mean, err, tau : ndarray
    """
    x = np.asarray(x, dtype=float)
    if tau is None:
        tau, _ = integrated_autocorrelation_time(x)
    n = x.shape[0]
    err = np.sqrt(2.0 * tau * x.var(axis=0) / n)
    return x.mean(axis=0), err, tau


def jackknife(func, *series, n_blocks=20):
    """
    Blocked jackknife estimate and error of ``func(*means)``.

    The series are cut into ``n_blocks`` contiguous blocks; ``func`` is
    evaluated on the sample means with one block left out at a time. Blocks
    should be much longer than the autocorrelation time.

    Parameters
    ----------
    func : callable
        Maps sample means of each series (each reduced over axis 0) to the
        derived quantity, e.g. ``lambda e, e2: e2 - e**2``.
    *series : array_like
        Samples, each shape ``(n, ...)`` with a common ``n``.
    n_blocks : int
        Number of jackknife blocks.

    Returns
    -------
    value, err : ndarray
        ``func`` of the full-sample means and its jackknife error.
    """
    series = [np.asarray(s, dtype=float) for s in series]
    n = series[0].shape[0]
    n_blocks = min(n_blocks, n)
    m = n // n_blocks * n_blocks
    if m < 2:
        value = func(*[s.mean(axis=0) for s in series])
        return value, np.full_like(np.asarray(value, dtype=float), np.nan)

    block_sums = [
        s[:m].reshape(n_blocks, m // n_blocks, *s.shape[1:]).sum(axis=1)
        for s in series
    ]
    totals = [b.sum(axis=0) for b in block_sums]

    value = func(*[s[:m].mean(axis=0) for s in series])
    leave_out = np.stack(
        [
            func(
                *[
                    (t - b[k]) / (m - m // n_blocks)
                    for t, b in zip(totals, block_sums)
                ]
            )
            for k in range(n_blocks)
        ]
    )
    err = np.sqrt(
        (n_blocks - 1) / n_blocks * ((leave_out - value) ** 2).sum(axis=0)
    )
    return value, err
