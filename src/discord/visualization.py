import colorsys

import numpy as np

import scipy.linalg
import scipy.signal

import pyvista as pv

import matplotlib.pyplot as plt

from matplotlib.transforms import Affine2D

from mpl_toolkits.axisartist import Axes, GridHelperCurveLinear
from mpl_toolkits.axisartist.grid_finder import (
    ExtremeFinderSimple,
    MaxNLocator,
)

from mantid.simpleapi import (
    CreateMDWorkspace,
    BinMD,
    IntegrateMDHistoWorkspace,
    SetMDFrame,
    SetUB,
    AddSampleLog,
    SaveMD,
    mtd,
)


class VisualizeReciprocalSpace:
    """Matplotlib-based 2D visualization of intensity slices in reciprocal space.

    Parameters
    ----------
    crystal : discord.material.Crystal
        Crystal instance providing lattice and supercell shape.
    """

    def __init__(self, crystal):
        self.crystal = crystal

    def get_projection_names(self, W):
        char_dict = {0: "0", 1: "{1}", -1: "-{1}"}
        chars = ["h", "k", "l"]
        names = [
            "["
            + ",".join(
                char_dict.get(j, "{0}{1}").format(
                    j, chars[np.argmax(np.abs(W[:, i]))]
                )
                for j in W[:, i]
            )
            + "]"
            for i in range(3)
        ]
        return names

    def get_projection_matrix(self):
        ei = mtd["volume"].getExperimentInfo(0)
        W = np.eye(3)
        if ei.run().hasProperty("W_MATRIX"):
            W = np.array(ei.run().getLogData("W_MATRIX").value).reshape(3, 3)
        return W

    def save(self, filename):
        SaveMD(InputWorkspace="volume", Filename=filename)

    def create_volume(
        self,
        x1=[-0.01, 0.01],
        x2=[-0.01, 0.01],
        x3=[-0.01, 0.01],
        n1=1,
        n2=1,
        n3=1,
        p1=[1, 0, 0],
        p2=[0, 1, 0],
        p3=[0, 0, 1],
    ):
        W = np.column_stack([p1, p2, p3])
        ax1, ax2, ax3 = self.get_projection_names(W)

        CreateMDWorkspace(
            Dimensions=3,
            Extents=[*x1, *x2, *x3],
            Names=[ax1, ax2, ax3],
            Units=3 * ["r.l.u."],
            OutputWorkspace="volume",
        )
        BinMD(
            InputWorkspace="volume",
            AlignedDim0=f"{ax1}, {x1[0]}, {x1[1]}, {n1}",
            AlignedDim1=f"{ax2}, {x2[0]}, {x2[1]}, {n2}",
            AlignedDim2=f"{ax3}, {x3[0]}, {x3[1]}, {n3}",
            OutputWorkspace="volume",
        )
        SetMDFrame(InputWorkspace="volume", MDFrame="HKL", Axes="0,1,2")

        W_MATRIX = ",".join(9 * ["{}"]).format(*W.flatten())
        AddSampleLog(
            Workspace="volume",
            LogName="W_MATRIX",
            LogText=W_MATRIX,
            LogType="String",
        )

        run = mtd["volume"].getExperimentInfo(0).run()
        run.addProperty("W_MATRIX", list(W.flatten() * 1.0), True)

        B = self.crystal.get_reciprocal_cartesian_transform()
        SetUB(Workspace="volume", UB=B)

        (x1, x2, x3), _, _ = self.extract_data()

        hkl = np.einsum("ij,j...->...i", W, [x1, x2, x3]).reshape(-1, 3)

        return hkl

    def update_result(self, I):
        dims = [
            mtd["volume"].getDimension(i)
            for i in range(mtd["volume"].getNumDims())
        ]
        n = [dim.getNBins() for dim in dims]
        mtd["volume"].setSignalArray(I.reshape(n))

    def extract_data(self):
        dims = [
            mtd["volume"].getDimension(i)
            for i in range(mtd["volume"].getNumDims())
        ]

        xs = np.meshgrid(
            *[
                np.linspace(
                    dim.getMinimum() + dim.getBinWidth() / 2,
                    dim.getMaximum() - dim.getBinWidth() / 2,
                    dim.getNBins(),
                )
                for dim in dims
            ],
            indexing="ij",
        )

        signal = mtd["volume"].getSignalArray().copy()
        errors = np.sqrt(mtd["volume"].getErrorSquaredArray())

        return xs, signal, errors

    def filter_intensity(self, m=2):
        N = np.array(self.crystal.get_super_cell_shape())
        r_cut = 0.5 * (1 - 1 / m)

        W = self.get_projection_matrix()

        (x1, x2, x3), I, _ = self.extract_data()
        hkl = np.einsum("ij,j...->...i", W, [x1, x2, x3])

        Q = hkl * N
        G = np.round(Q) * 0

        l = np.sinc((Q - G) / m)
        l[np.abs(Q - G) > m] = 0.0

        L = np.prod(l, axis=3)
        W_sinc = np.prod(np.sinc(r_cut * (Q - G)), axis=3)

        I[~np.isfinite(I)] = 0

        W_Q = L * W_sinc
        W_Q /= np.sum(W_Q)

        I_filt = scipy.signal.convolve(I, W_Q, mode="same")

        mtd["volume"].setSignalArray(I_filt)

    def plot_slice(
        self, value=0, thickness=0.1, normal=[0, 0, 1], filename=None
    ):
        self.filter_intensity()
        W = self.get_projection_matrix()
        B = self.crystal.get_reciprocal_cartesian_transform()

        integrate = [value - thickness, value + thickness]
        slice_lims = [None, None]

        pbin = []
        j = 0
        for i, norm in enumerate(normal):
            if norm == 0:
                pbin.append(slice_lims[j])
                j += 1
            else:
                pbin.append(integrate)

        IntegrateMDHistoWorkspace(
            InputWorkspace="volume",
            P1Bin=pbin[0],
            P2Bin=pbin[1],
            P3Bin=pbin[2],
            OutputWorkspace="slice",
        )

        i = np.abs(normal).tolist().index(1)

        form = "{} = ({:.2f},{:.2f})"

        title = form.format(mtd["slice"].getDimension(i).name, *integrate)
        dims = mtd["slice"].getNonIntegratedDimensions()

        x, y = [
            np.linspace(
                dim.getMinimum(), dim.getMaximum(), dim.getNBoundaries()
            )
            for dim in dims
        ]

        labels = ["{} ({})".format(dim.name, dim.getUnits()) for dim in dims]
        signal = mtd["slice"].getSignalArray().T.copy().squeeze()

        signal[signal <= 0] = np.nan
        signal[np.isinf(signal)] = np.nan

        Bp = np.dot(B, W)

        Q, R = scipy.linalg.qr(Bp)

        ind = np.abs(normal) != 1
        i = ind.tolist().index(False)

        v = scipy.linalg.cholesky(np.dot(R.T, R)[ind][:, ind], lower=False)

        v /= v[0, 0]

        T = np.eye(3)
        T[:2, :2] = v

        s = np.diag(T).copy()
        T[1, 1] = 1

        T[0, 2] = -T[0, 1] * y.min()

        aspect = s[1]

        transform = Affine2D(T)

        extreme_finder = ExtremeFinderSimple(20, 20)
        grid_locator1 = MaxNLocator(nbins=10)
        grid_locator2 = MaxNLocator(nbins=10)
        grid_locator1.set_params(integer=True)
        grid_locator2.set_params(integer=True)

        grid_helper = GridHelperCurveLinear(
            transform,
            extreme_finder=extreme_finder,
            grid_locator1=grid_locator1,
            grid_locator2=grid_locator2,
        )

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, axes_class=Axes, grid_helper=grid_helper)
        ax.minorticks_on()

        ax.set_aspect(aspect)
        trans = transform + ax.transData

        im = ax.pcolormesh(
            x,
            y,
            signal,
            shading="flat",
            transform=trans,
            rasterized=True,
        )

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

        ax.set_title(title)

        cb = fig.colorbar(im, ax=ax)
        cb.minorticks_on()

        fig.savefig(filename)


class VisualizeCorrelations:
    """Matplotlib-based 2D visualization of real-space spin correlations.

    Parameters
    ----------
    crystal : discord.material.Crystal
        Crystal instance providing lattice and supercell shape.
    """

    def __init__(self, crystal):
        self.crystal = crystal

    def get_projection_names(self, W):
        char_dict = {0: "0", 1: "{1}", -1: "-{1}"}
        chars = ["u", "v", "w"]
        names = [
            "["
            + ",".join(
                char_dict.get(j, "{0}{1}").format(
                    j, chars[np.argmax(np.abs(W[:, i]))]
                )
                for j in W[:, i]
            )
            + "]"
            for i in range(3)
        ]
        return names

    def update_result(self, C_ij):
        self.C_ij = C_ij

    def plot_slice(
        self,
        p1=[1, 0, 0],
        p2=[0, 1, 0],
        p3=[0, 0, 1],
        value=0,
        tolerance=0.1,
        filename=None,
    ):
        """Plot in-plane correlations summed over all atom pairs.

        Parameters
        ----------
        filename : str
            Output image filename.
        """

        n_atoms, _, nx, ny, nz = self.C_ij.shape

        xyz = self.crystal.get_unit_atom_position()
        A = self.crystal.get_direct_cartesian_transform()

        Cshift = np.fft.fftshift(self.C_ij, axes=(2, 3, 4))

        dx = np.arange(nx) - nx // 2
        dy = np.arange(ny) - ny // 2
        dz = np.arange(nz) - nz // 2

        Dx, Dy, Dz = np.meshgrid(dx, dy, dz, indexing="ij")

        i_idx, j_idx = np.meshgrid(
            np.arange(n_atoms), np.arange(n_atoms), indexing="ij"
        )
        i_idx = i_idx.reshape(-1)
        j_idx = j_idx.reshape(-1)

        d_xyz = xyz[j_idx] - xyz[i_idx]

        corr_xy = Cshift[i_idx, j_idx, :, :, :]
        corr_flat = corr_xy.reshape(-1, nx * ny * nz)

        r_frac = np.column_stack([Dx.ravel(), Dy.ravel(), Dz.ravel()])
        r_frac = r_frac[None, :, :] + d_xyz[:, None, :]

        x = r_frac[..., 0]
        y = r_frac[..., 1]
        z = r_frac[..., 2]

        W = np.column_stack([p1, p2, p3])

        xlabel, ylabel, title = self.get_projection_names(W)

        x1 = p1[0] * x + p1[1] * y + p1[2] * z
        x2 = p2[0] * x + p2[1] * y + p2[2] * z
        x3 = p3[0] * x + p3[1] * y + p3[2] * z

        X1 = p1[0] * Dx + p1[1] * Dy + p1[2] * Dz
        X2 = p2[0] * Dx + p2[1] * Dy + p2[2] * Dz
        X3 = p3[0] * Dx + p3[1] * Dy + p3[2] * Dz

        condition = x3 - value

        mask = np.isclose(condition, 0.0, atol=tolerance)
        r1_all = x1[mask]
        r2_all = x2[mask]
        c_all = corr_flat[mask]

        if r1_all.size > 0:
            key = np.round(np.column_stack([r1_all, r2_all]), decimals=8)
            uniq, inv = np.unique(key, axis=0, return_inverse=True)
            n_unique = uniq.shape[0]

            r1 = uniq[:, 0]
            r2 = uniq[:, 1]
            c_sum = np.bincount(inv, weights=c_all, minlength=n_unique)
            counts = np.bincount(inv, minlength=n_unique)
            c = c_sum / counts
        else:
            r1 = r2 = c = np.array([])

        Ap = np.dot(A, W)

        Q, R = scipy.linalg.qr(Ap)

        v = scipy.linalg.cholesky(np.dot(R.T, R)[:2, :2], lower=False)
        v /= v[0, 0]

        T = np.eye(3)
        T[:2, :2] = v

        scale = np.diag(T).copy()
        T[1, 1] = 1.0

        T[0, 2] = -T[0, 1] * x2.min()

        aspect = scale[1]

        transform = Affine2D(T)

        extreme_finder = ExtremeFinderSimple(20, 20)
        grid_locator1 = MaxNLocator(nbins=10)
        grid_locator2 = MaxNLocator(nbins=10)
        grid_locator1.set_params(integer=True)
        grid_locator2.set_params(integer=True)

        grid_helper = GridHelperCurveLinear(
            transform,
            extreme_finder=extreme_finder,
            grid_locator1=grid_locator1,
            grid_locator2=grid_locator2,
        )

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, axes_class=Axes, grid_helper=grid_helper)

        trans = transform + ax.transData

        ax.minorticks_on()

        sc = ax.scatter(
            r1,
            r2,
            c=c,
            cmap="seismic",
            vmin=-1,
            vmax=1,
            transform=trans,
            rasterized=True,
        )

        ax.set_aspect(aspect)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        label = (
            r"$\langle \boldsymbol{S}_i(0) \cdot \boldsymbol{S}_j(r) \rangle$"
        )

        cb = fig.colorbar(sc, label=label)
        cb.ax.minorticks_on()

        super_mask = np.isclose(X3 - value, 0.0, atol=tolerance)

        if np.any(super_mask):
            xmin = X1[super_mask].min()
            xmax = X1[super_mask].max()
            ymin = X2[super_mask].min()
            ymax = X2[super_mask].max()

            bx = [xmin, xmax, xmax, xmin, xmin]
            by = [ymin, ymin, ymax, ymax, ymin]

            ax.plot(bx, by, color="k", linewidth=1, transform=trans)

        fig.savefig(filename, dpi=200)


class VisualizeAtoms:
    """PyVista-based 3D visualization of atom configurations in a supercell.

    Parameters
    ----------
    crystal : discord.material.Crystal
        Crystal instance providing lattice, supercell shape, and atom positions.
    """

    def __init__(self, crystal):
        self.crystal = crystal

        self.plotter = pv.Plotter()
        self.show_axes()

    def show_axes(self):
        t = pv._vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                t.SetElement(i, j, self.crystal.C[i, j])
        actor = self.plotter.add_axes(xlabel="a", ylabel="b", zlabel="c")
        actor.SetUserMatrix(t)

    def plot_spins(
        self,
        spins,
        filename=None,
        glyph_length=1.0,
        line_opacity=0.1,
    ):
        """Render a 3D spin configuration and save a screenshot.

        Parameters
        ----------
        spins : array_like
            Spin vectors with shape ``(n_atoms, ni, nj, nk, 3)`` or compatible.
        filename : str, optional
            Output image filename.
        window_size : tuple[int, int], optional
            Size of the PyVista window in pixels.
        glyph_length : float, optional
            Length of arrow glyphs representing spins.
        line_opacity : float, optional
            Opacity for the supercell grid lines.
        """
        s = np.asarray(spins)
        r = self.crystal.get_atom_positions()

        points = r.reshape(-1, 3)
        vectors = s.reshape(-1, 3)

        phi = np.arctan2(vectors[:, 1], vectors[:, 0])
        hue = (phi + np.pi) / (2 * np.pi)
        lightness = 0.15 + 0.7 * (vectors[:, 2] + 1) / 2.0
        saturation = np.ones_like(hue)

        rgb = np.array(
            [
                colorsys.hls_to_rgb(hh, ll, ss)
                for hh, ll, ss in zip(hue, lightness, saturation)
            ]
        )
        rgb_255 = (255 * np.clip(rgb, 0, 1)).astype(np.uint8)

        vec_norm = np.linalg.norm(vectors, axis=1)
        vec_norm[vec_norm == 0] = 1.0
        unit_vec = vectors / vec_norm[:, None]

        offset = 0.5 * glyph_length
        centered_points = points - offset * unit_vec

        cloud = pv.PolyData(centered_points)
        cloud["vectors"] = unit_vec
        cloud["colors"] = rgb_255

        arrow = pv.Arrow(scale=1.0)
        glyphs = cloud.glyph(
            orient="vectors", scale=False, factor=glyph_length, geom=arrow
        )

        self.plotter.add_mesh(glyphs, scalars="colors", rgb=True)

        A = self.crystal.get_direct_cartesian_transform()
        ni, nj, nk = self.crystal.get_super_cell_shape()

        def frac_to_cart(i, j, k):
            return A @ np.array([i, j, k], dtype=float)

        for j in range(nj + 1):
            for k in range(nk + 1):
                p0 = frac_to_cart(0, j, k)
                p1 = frac_to_cart(ni, j, k)
                self.plotter.add_mesh(
                    pv.Line(p0, p1),
                    color="black",
                    line_width=1,
                    opacity=line_opacity,
                )

        for i in range(ni + 1):
            for k in range(nk + 1):
                p0 = frac_to_cart(i, 0, k)
                p1 = frac_to_cart(i, nj, k)
                self.plotter.add_mesh(
                    pv.Line(p0, p1),
                    color="black",
                    line_width=1,
                    opacity=line_opacity,
                )

        for i in range(ni + 1):
            for j in range(nj + 1):
                p0 = frac_to_cart(i, j, 0)
                p1 = frac_to_cart(i, j, nk)
                self.plotter.add_mesh(
                    pv.Line(p0, p1),
                    color="black",
                    line_width=1,
                    opacity=line_opacity,
                )

        self.plotter.show(screenshot=filename)
        self.plotter.close()
