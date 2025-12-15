import colorsys

import numpy as np

import pyvista as pv


class Visualize:
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
        filename="spin_configuration.png",
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
