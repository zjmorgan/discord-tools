import numpy as np
import pytest

from discord.atomistic.order import ordering_modes, order_parameter

N = (4, 4, 4)


def grid():
    return np.stack(np.meshgrid(*[np.arange(n) for n in N], indexing="ij"))


def test_ferromagnet():
    mu = np.zeros((1, *N, 3))
    mu[..., 2] = 2.0
    q, modes, _ = ordering_modes(mu)
    assert q.tolist() == [[0, 0, 0]]
    assert order_parameter(mu, q, modes) == pytest.approx(2.0)


def test_checkerboard_antiferromagnet():
    i, j, k = grid()
    mu = np.zeros((1, *N, 3))
    mu[0, ..., 0] = (-1.0) ** (i + j + k)
    q, modes, _ = ordering_modes(mu)
    assert q.tolist() == [[2, 2, 2]]  # (1/2, 1/2, 1/2)
    assert order_parameter(mu, q, modes) == pytest.approx(1.0)


def test_two_sublattice_antiferromagnet_at_zero_wavevector():
    # Rutile-like: two atoms per cell with opposite moments; net M = 0.
    mu = np.zeros((2, *N, 3))
    mu[0, ..., 2] = 1.5
    mu[1, ..., 2] = -1.5
    q, modes, _ = ordering_modes(mu)
    assert q.tolist() == [[0, 0, 0]]
    assert np.allclose(np.abs(modes[0]), 1 / np.sqrt(2))
    assert np.real(modes[0, 0] * np.conj(modes[0, 1])) < 0
    assert order_parameter(mu, q, modes) == pytest.approx(1.5)


def test_helix_uses_both_wavevectors():
    i, _, _ = grid()
    phase = 2 * np.pi * i / 4
    mu = np.zeros((1, *N, 3))
    mu[0, ..., 0] = np.cos(phase)
    mu[0, ..., 1] = np.sin(phase)
    q, modes, _ = ordering_modes(mu)
    assert sorted(q.tolist()) == [[1, 0, 0], [3, 0, 0]]  # +q and -q
    assert order_parameter(mu, q, modes) == pytest.approx(1.0)


def test_equivalent_domains_share_the_order_parameter():
    # Stripes along x in one configuration and along y in another: the star
    # contains both wavevectors and each domain gives m = 1.
    i, j, _ = grid()
    a = np.zeros((1, *N, 3))
    b = np.zeros((1, *N, 3))
    a[0, ..., 2] = (-1.0) ** i
    b[0, ..., 2] = (-1.0) ** j
    q, modes, _ = ordering_modes(np.stack([a, b]))
    assert sorted(q.tolist()) == [[0, 2, 0], [2, 0, 0]]
    assert np.allclose(order_parameter(np.stack([a, b]), q, modes), 1.0)


def test_random_configuration_is_disordered():
    rng = np.random.default_rng(0)
    big = (2, 8, 8, 8)
    s = rng.normal(size=(*big, 3))
    s /= np.linalg.norm(s, axis=-1)[..., None]
    q = np.array([[0, 0, 0]])
    modes = np.array([[1, -1]]) / np.sqrt(2)
    # |m| ~ 1/sqrt(N) for random unit vectors
    assert order_parameter(s, q, modes) < 5 / np.sqrt(s[..., 0].size)
