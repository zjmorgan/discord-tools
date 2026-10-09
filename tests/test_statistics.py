import numpy as np
import pytest

from discord.atomistic import statistics


def ar1(phi, n, n_series=1, seed=0):
    """AR(1) process x_t = phi x_{t-1} + noise, started in equilibrium."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n, n_series))
    x = np.empty((n, n_series))
    x[0] = noise[0] / np.sqrt(1.0 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    return x


def test_autocorrelation_ar1():
    phi = 0.8
    rho = statistics.autocorrelation(ar1(phi, 200_000))[:, 0]
    assert rho[0] == pytest.approx(1.0)
    assert np.allclose(rho[1:6], phi ** np.arange(1, 6), atol=0.01)


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9, 0.97])
def test_tau_int_ar1(phi):
    # Exact: tau_int = 1/2 + sum_t phi^t = (1 + phi) / (2 (1 - phi))
    exact = (1.0 + phi) / (2.0 * (1.0 - phi))
    x = ar1(phi, 400_000, n_series=4, seed=1)
    tau, window = statistics.integrated_autocorrelation_time(x)
    assert tau.shape == (4,)
    assert np.all(window >= 1)
    assert np.allclose(tau, exact, rtol=0.1)


def test_tau_int_trailing_axes_and_constant_series():
    x = ar1(0.5, 10_000, n_series=6, seed=2).reshape(10_000, 2, 3)
    x[:, 1, 2] = 3.0
    tau, _ = statistics.integrated_autocorrelation_time(x)
    assert tau.shape == (2, 3)
    assert tau[1, 2] == 0.5
    assert np.all(np.isfinite(tau))


def test_tau_int_short_series():
    tau, window = statistics.integrated_autocorrelation_time(np.ones((2, 3)))
    assert np.all(tau == 0.5)
    assert np.all(window == 1)


def test_mean_error_coverage():
    # Over many independent correlated chains, the corrected error should
    # match the actual scatter of the means (naive error is ~4x too small).
    phi = 0.9
    x = ar1(phi, 20_000, n_series=200, seed=3)
    mean, err, tau = statistics.mean_error(x)
    ratio = mean.std() / err.mean()
    assert 0.85 < ratio < 1.15

    naive = x.std(axis=0) / np.sqrt(len(x))
    assert mean.std() / naive.mean() > 3.5


def test_jackknife_variance_of_gaussian():
    # Standard error of the sample variance: sigma^2 * sqrt(2 / (n - 1))
    rng = np.random.default_rng(4)
    sigma, n = 2.0, 50_000
    x = rng.normal(0.0, sigma, size=n)

    value, err = statistics.jackknife(
        lambda m, m2: m2 - m**2, x, x**2, n_blocks=50
    )
    assert value == pytest.approx(x.var(), rel=1e-12)
    assert err == pytest.approx(sigma**2 * np.sqrt(2.0 / (n - 1)), rel=0.3)


def test_jackknife_vector_valued_func():
    rng = np.random.default_rng(5)
    m = rng.normal(size=(5000, 3))
    mm = m[:, :, None] * m[:, None, :]
    value, err = statistics.jackknife(
        lambda a, b: b - np.outer(a, a), m, mm, n_blocks=20
    )
    assert value.shape == (3, 3) and err.shape == (3, 3)
    assert np.allclose(value, np.eye(3), atol=0.1)
    assert np.all(err > 0)
