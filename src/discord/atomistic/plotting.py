"""Plotting utilities for Monte Carlo simulation results."""

import os

import numpy as np
import matplotlib.pyplot as plt


def plot_susceptibility(T, chi, filename=None, show=False):
    """
    Plot magnetic susceptibility tensor components.

    Parameters
    ----------
    T : array_like
        Temperature array
    chi : array_like
        Susceptibility tensor with shape (n_temps, 3, 3)
    filename : str, optional
        If provided, save figure to this path
    show : bool, optional
        If True, display the figure

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    chi_11 = chi[:, 0, 0]
    chi_22 = chi[:, 1, 1]
    chi_33 = chi[:, 2, 2]
    chi_23 = chi[:, 1, 2]
    chi_13 = chi[:, 0, 2]
    chi_12 = chi[:, 0, 1]

    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(8, 6))
    ax.minorticks_on()
    ax.plot(T, chi_11, "-o", label="$\chi_{11}$")
    ax.plot(T, chi_22, "-o", label="$\chi_{22}$")
    ax.plot(T, chi_33, "-o", label="$\chi_{33}$")
    ax.plot(T, chi_23, "-o", label="$\chi_{23}$")
    ax.plot(T, chi_13, "-o", label="$\chi_{13}$")
    ax.plot(T, chi_12, "-o", label="$\chi_{12}$")
    ax.legend(shadow=True)
    ax.set_xlabel("$T$ [K]")
    ax.set_ylabel("$\chi_{ij}$ [$\mu_B^2$/eV]")
    ax.grid(alpha=0.3)

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_magnetization(T, M_ave, M_std=None, filename=None, show=False):
    """
    Plot magnetization components.

    Parameters
    ----------
    T : array_like
        Temperature array
    M_ave : array_like
        Average magnetization with shape (n_temps, 3)
    M_std : array_like, optional
        Standard deviation of magnetization
    filename : str, optional
        If provided, save figure to this path
    show : bool, optional
        If True, display the figure

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    Mx = M_ave[:, 0]
    My = M_ave[:, 1]
    Mz = M_ave[:, 2]

    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(8, 6))
    ax.minorticks_on()

    if M_std is not None:
        Mx_std = M_std[:, 0]
        My_std = M_std[:, 1]
        Mz_std = M_std[:, 2]
        ax.errorbar(T, Mx, Mx_std, fmt="-o", label="$M_{x}$", capsize=3)
        ax.errorbar(T, My, My_std, fmt="-o", label="$M_{y}$", capsize=3)
        ax.errorbar(T, Mz, Mz_std, fmt="-o", label="$M_{z}$", capsize=3)
    else:
        ax.plot(T, Mx, "-o", label="$M_{x}$")
        ax.plot(T, My, "-o", label="$M_{y}$")
        ax.plot(T, Mz, "-o", label="$M_{z}$")

    ax.legend(shadow=True)
    ax.set_xlabel("$T$ [K]")
    ax.set_ylabel("$M_{i}$ [$\mu_B$]")
    ax.grid(alpha=0.3)

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_heat_capacity(T, C, filename=None, show=False):
    """
    Plot heat capacity.

    Parameters
    ----------
    T : array_like
        Temperature array
    C : array_like
        Heat capacity
    filename : str, optional
        If provided, save figure to this path
    show : bool, optional
        If True, display the figure

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(8, 6))
    ax.minorticks_on()
    ax.plot(T, C, "-o")
    ax.set_xlabel("$T$ [K]")
    ax.set_ylabel("$C$ [eV/K]")
    ax.grid(alpha=0.3)

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_energy(T, E, E_std=None, filename=None, show=False):
    """
    Plot energy.

    Parameters
    ----------
    T : array_like
        Temperature array
    E : array_like
        Average energy
    E_std : array_like, optional
        Standard deviation of energy
    filename : str, optional
        If provided, save figure to this path
    show : bool, optional
        If True, display the figure

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(8, 6))
    ax.minorticks_on()

    if E_std is not None:
        ax.errorbar(T, E, E_std, fmt="-o", capsize=3)
    else:
        ax.plot(T, E, "-o")

    ax.set_xlabel("$T$ [K]")
    ax.set_ylabel("$E$ [eV]")
    ax.grid(alpha=0.3)

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_intensity(T, I, sig=None, hkl=None, filename=None, show=False):
    """
    Plot magnetic Bragg intensity.

    Parameters
    ----------
    T : array_like
        Temperature array
    I : array_like
        Intensity
    sig : array_like, optional
        Standard deviation of intensity
    hkl : array_like, optional
        HKL indices for title
    filename : str, optional
        If provided, save figure to this path
    show : bool, optional
        If True, display the figure

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(8, 6))
    ax.minorticks_on()

    if sig is not None:
        ax.errorbar(T, I, sig, fmt="-o", capsize=3)
    else:
        ax.plot(T, I, "-o")

    ax.set_xlabel("$T$ [K]")
    ax.set_ylabel("$I$ [arb. units]")
    ax.grid(alpha=0.3)

    if hkl is not None:
        ax.set_title(f"({hkl[0]:.0f} {hkl[1]:.0f} {hkl[2]:.0f})")

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_results(result, prefix="mc", outdir=".", show=False):
    """
    Generate all standard plots from simulation results.

    Parameters
    ----------
    result : dict
        Dictionary returned by parallel_tempering
    prefix : str, optional
        Prefix for output filenames
    outdir : str, optional
        Directory to save plots
    show : bool, optional
        If True, display figures

    Returns
    -------
    figures : dict
        Dictionary of matplotlib figures
    """
    os.makedirs(outdir, exist_ok=True)

    T = result["T"]
    figures = {}

    # Susceptibility
    fig, _ = plot_susceptibility(
        T,
        result["chi"],
        filename=os.path.join(outdir, prefix + "_susceptibility.png"),
        show=show,
    )
    figures["susceptibility"] = fig

    # Magnetization
    fig, _ = plot_magnetization(
        T,
        result["M(ave)"],
        result["M(std)"],
        filename=os.path.join(outdir, prefix + "_magnetization.png"),
        show=show,
    )
    figures["magnetization"] = fig

    # Heat capacity
    fig, _ = plot_heat_capacity(
        T,
        result["C"],
        filename=os.path.join(outdir, prefix + "_heat_capacity.png"),
        show=show,
    )
    figures["heat_capacity"] = fig

    # Energy
    fig, _ = plot_energy(
        T,
        result["E(ave)"],
        result["E(std)"],
        filename=os.path.join(outdir, prefix + "_energy.png"),
        show=show,
    )
    figures["energy"] = fig

    # Intensity (if available)
    if result["I(ave)"] is not None:
        I = result["I(ave)"][:, 0]
        sig = result["I(std)"][:, 0]
        fig, _ = plot_intensity(
            T,
            I,
            sig,
            filename=os.path.join(outdir, prefix + "_intensity.png"),
            show=show,
        )
        figures["intensity"] = fig

    return figures
