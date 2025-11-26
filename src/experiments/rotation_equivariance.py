# rotation equivariance experiments

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
from scipy.spatial.transform import Rotation
from matplotlib.colors import LightSource
from matplotlib import tri as mtri

from ..config import (
    SEED,
    RESULTS_DIR,
    FIGURE_DPI,
    SO3_PLANE_RES,
    SO3_PLANE_SIZE,
    SO3_N_BINS,
    SO3_WIREFRAME,
    SO3_WIREFRAME_RSTRIDE,
    SO3_WIREFRAME_CSTRIDE,
    SO3_MESH_U,
    SO3_MESH_V,
    SO3_ENABLE_LIGHTING,
    SO3_USE_TRISURF,
    SO3_AXIS_LINEWIDTH,
    SO3_BACKDROP_ENABLE,
    SO3_BACKDROP_OFFSET,
    SO3_BACKDROP_COLOR,
    SO3_BACKDROP_ALPHA,
    SO3_BACKDROP_RSTRIDE,
    SO3_BACKDROP_CSTRIDE,
    SO3_BACKDROP_TICKS,
)
from ..utils import set_seed, save_figure


def create_2d_siren_field(x, y, seed=None):
    # 2d siren-like field
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    n_waves = 5
    field = np.zeros_like(x)
    
    for _ in range(n_waves):
        freq_x = rng.uniform(1, 4)
        freq_y = rng.uniform(1, 4)
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 1.5)
        field += amp * np.sin(freq_x * x + phase_x) * np.cos(freq_y * y + phase_y)
    
    return field


def create_spherical_harmonic_field(theta, phi, seed=None):
    # spherical harmonic field
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    field = np.zeros_like(theta, dtype=np.float64)
    modes = [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1), (3, 2)]
    
    for l, m in modes:
        weight = rng.uniform(-1, 1)
        Y_lm = sph_harm(m, l, phi, theta)
        field += weight * np.real(Y_lm)
    
    return field


def rotation_matrix_2d(angle_deg):
    # 2d rotation matrix
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def run_so2_equivariance(n_grid=64, extent=1.5, angles=None, seed=None):
    # run so(2) rotation test
    if seed is None:
        seed = SEED
    if angles is None:
        angles = [0, 45, 90, 135]
    
    print("\n  running so(2) equivariance test...")
    
    x = np.linspace(-extent, extent, n_grid)
    y = np.linspace(-extent, extent, n_grid)
    xx, yy = np.meshgrid(x, y)
    
    fields = []
    
    for angle in angles:
        R = rotation_matrix_2d(-angle)
        coords = np.stack([xx.flatten(), yy.flatten()], axis=1)
        coords_rot = coords @ R.T
        xx_rot = coords_rot[:, 0].reshape(n_grid, n_grid)
        yy_rot = coords_rot[:, 1].reshape(n_grid, n_grid)
        
        field_rotated = create_2d_siren_field(xx_rot, yy_rot, seed=seed)
        fields.append(field_rotated)
    
    return {
        'angles': angles,
        'fields': fields,
        'extent': extent,
        'n_grid': n_grid
    }


def run_so3_equivariance(n_points=60, seed=None):
    # run so(3) rotation test
    if seed is None:
        seed = SEED
    
    print("\n  running so(3) equivariance test...")
    
    # use higher-resolution mesh from config for smooth sphere
    # longitude: exclude endpoint to avoid duplicated seam column so wireframes connect
    u = np.linspace(0, 2 * np.pi, SO3_MESH_U, endpoint=False)
    v = np.linspace(0, np.pi, SO3_MESH_V)
    u_grid, v_grid = np.meshgrid(u, v)
    
    x = np.sin(v_grid) * np.cos(u_grid)
    y = np.sin(v_grid) * np.sin(u_grid)
    z = np.cos(v_grid)
    
    field_base = create_spherical_harmonic_field(v_grid, u_grid, seed=seed)
    
    configs = [
        ('Identity', None),
        ('X-axis 45°', ('x', 45)),
        ('Y-axis 45°', ('y', 45)),
        ('Z-axis 45°', ('z', 45)),
        ('XY 30°', [('x', 30), ('y', 30)]),
        ('XYZ 30°', [('x', 30), ('y', 30), ('z', 30)])
    ]
    
    results = []
    for name, rot_spec in configs:
        if rot_spec is None:
            R = np.eye(3)
        elif isinstance(rot_spec, tuple):
            axis, angle = rot_spec
            R = Rotation.from_euler(axis, angle, degrees=True).as_matrix()
        else:
            R = np.eye(3)
            for axis, angle in rot_spec:
                R = Rotation.from_euler(axis, angle, degrees=True).as_matrix() @ R
        
        coords = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        coords_rot = coords @ R.T
        
        x_rot = coords_rot[:, 0].reshape(v_grid.shape)
        y_rot = coords_rot[:, 1].reshape(v_grid.shape)
        z_rot = coords_rot[:, 2].reshape(v_grid.shape)
        
        results.append({
            'name': name,
            'x': x_rot,
            'y': y_rot,
            'z': z_rot,
            'field': field_base,
            'R': R
        })
    
    return results


def create_so2_figure(results, verbose=True):
    # create so(2) figure - single row with rotated fields
    if verbose:
        print("\n  creating so(2) equivariance figure...")
    
    angles = results['angles']
    fields = results['fields']
    extent = results['extent']
    n_panels = len(angles)
    
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3.5))
    if n_panels == 1:
        axes = [axes]
    
    vmin = min(f.min() for f in fields)
    vmax = max(f.max() for f in fields)
    vabs = max(abs(vmin), abs(vmax))
    
    n_contours = 8
    
    for i, angle in enumerate(angles):
        ax = axes[i]
        field = fields[i]
        
        im = ax.imshow(field, extent=[-extent, extent, -extent, extent],
                      origin='lower', cmap='RdBu_r', vmin=-vabs, vmax=vabs)
        
        # contour lines (level sets)
        x_cont = np.linspace(-extent, extent, field.shape[1])
        y_cont = np.linspace(-extent, extent, field.shape[0])
        ax.contour(x_cont, y_cont, field, levels=n_contours, 
                  colors='black', linewidths=0.5, alpha=0.6)
        
        # tracking arrow
        arrow_r = extent * 0.55
        arrow_angle = np.radians(angle + 45)
        ax.annotate('', 
                   xy=(arrow_r * np.cos(arrow_angle), arrow_r * np.sin(arrow_angle)),
                   xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2))
        
        ax.set_title(f'θ = {angle}°', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    
    # colorbar on far right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Field Value', fontsize=10)
    
    plt.subplots_adjust(left=0.03, right=0.90, top=0.92, bottom=0.05, wspace=0.08)
    save_figure(fig, 'so2_rotation_equivariance')
    plt.close(fig)


def create_so3_figure(results, verbose=True):
    # create so(3) figure - spheres with xyz coordinate planes overlaid
    if verbose:
        print("\n  creating so(3) equivariance figure...")
    
    n_panels = len(results)
    n_cols = 3
    n_rows = 2
    
    fig = plt.figure(figsize=(5.5 * n_cols, 5 * n_rows))
    
    all_fields = [r['field'] for r in results]
    vmin = min(f.min() for f in all_fields)
    vmax = max(f.max() for f in all_fields)
    vabs = max(abs(vmin), abs(vmax))
    
    for idx, result in enumerate(results):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        
        x, y, z = result['x'], result['y'], result['z']
        field = result['field']
        R = result['R']

        # draw fixed backdrop grids (orthogonal) behind the sphere so the backdrop isn't blank
        if SO3_BACKDROP_ENABLE:
            # create a grid in the xy-plane at constant z = offset
            p = np.linspace(-SO3_PLANE_SIZE, SO3_PLANE_SIZE, SO3_PLANE_RES)
            p1, p2 = np.meshgrid(p, p)
            # XY grid at z = offset
            z_plane = np.full_like(p1, SO3_BACKDROP_OFFSET)
            ax.plot_wireframe(p1, p2, z_plane, rstride=SO3_BACKDROP_RSTRIDE, cstride=SO3_BACKDROP_CSTRIDE,
                              color=SO3_BACKDROP_COLOR, linewidth=0.5, alpha=SO3_BACKDROP_ALPHA)
            # YZ grid at x = offset
            x_plane = np.full_like(p1, SO3_BACKDROP_OFFSET)
            ax.plot_wireframe(x_plane, p1, p2, rstride=SO3_BACKDROP_RSTRIDE, cstride=SO3_BACKDROP_CSTRIDE,
                              color=SO3_BACKDROP_COLOR, linewidth=0.5, alpha=SO3_BACKDROP_ALPHA)
            # XZ grid at y = offset
            y_plane = np.full_like(p1, SO3_BACKDROP_OFFSET)
            ax.plot_wireframe(p1, y_plane, p2, rstride=SO3_BACKDROP_RSTRIDE, cstride=SO3_BACKDROP_CSTRIDE,
                              color=SO3_BACKDROP_COLOR, linewidth=0.5, alpha=SO3_BACKDROP_ALPHA)

            # numeric tick labels on the backdrop grids (so grids are informative)
            ticks = np.linspace(-SO3_PLANE_SIZE, SO3_PLANE_SIZE, SO3_BACKDROP_TICKS)
            # XY plane labels (z = offset): label X along +x edge and Y along +y edge
            for t in ticks:
                # X ticks along x-axis at y = min, z = offset
                ax.text(t, -SO3_PLANE_SIZE - 0.02, SO3_BACKDROP_OFFSET - 0.01, f"{t:.2f}",
                        color=SO3_BACKDROP_COLOR, fontsize=7, ha='center', va='top')
                # Y ticks along y-axis at x = min
                ax.text(-SO3_PLANE_SIZE - 0.02, t, SO3_BACKDROP_OFFSET - 0.01, f"{t:.2f}",
                        color=SO3_BACKDROP_COLOR, fontsize=7, ha='right', va='center')

            # YZ plane labels (x = offset): label Y and Z
            for t in ticks:
                ax.text(SO3_BACKDROP_OFFSET - 0.01, t, -SO3_PLANE_SIZE - 0.02, f"{t:.2f}",
                        color=SO3_BACKDROP_COLOR, fontsize=7, ha='center', va='top')
                ax.text(SO3_BACKDROP_OFFSET - 0.01, -SO3_PLANE_SIZE - 0.02, t, f"{t:.2f}",
                        color=SO3_BACKDROP_COLOR, fontsize=7, ha='right', va='center')

            # XZ plane labels (y = offset): label X and Z
            for t in ticks:
                ax.text(t, SO3_BACKDROP_OFFSET - 0.01, -SO3_PLANE_SIZE - 0.02, f"{t:.2f}",
                        color=SO3_BACKDROP_COLOR, fontsize=7, ha='center', va='top')
                ax.text(-SO3_PLANE_SIZE - 0.02, SO3_BACKDROP_OFFSET - 0.01, t, f"{t:.2f}",
                        color=SO3_BACKDROP_COLOR, fontsize=7, ha='right', va='center')

        # prepare coloring: use continuous RdBu_r and LightSource shading for 3D effect
        cmap = plt.cm.RdBu_r
        if SO3_ENABLE_LIGHTING:
            ls = LightSource(azdeg=315, altdeg=45)
            # shade returns rgb array compatible with facecolors
            rgb = ls.shade(field, cmap=cmap, vert_exag=1, blend_mode='soft')
        else:
            # fallback to plain colormap mapping
            field_norm = (field + vabs) / (2 * vabs)
            field_norm = np.clip(field_norm, 0, 1)
            rgb = cmap(field_norm)

        # Plot the surface with full resolution (rstride=1, cstride=1) and antialiasing
        try_trisurf = SO3_USE_TRISURF
        if try_trisurf:
            # create 2D triangulation in parameter space and map to 3D
            u_flat = np.mod(np.arctan2(y, x) + 2 * np.pi, 2 * np.pi).flatten()
            v_flat = np.arccos(z).flatten()
            tri = mtri.Triangulation(u_flat, v_flat)
            ax.plot_trisurf(x.flatten(), y.flatten(), z.flatten(), triangles=tri.triangles,
                            linewidth=0, antialiased=True, shade=True, cmap=cmap, alpha=1.0)
        else:
            ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                            linewidth=0, antialiased=True, shade=True, alpha=1.0)

        # overlay a subtle black wireframe to make the spherical grid visible (no planes)
        if SO3_WIREFRAME:
            ax.plot_wireframe(x, y, z, rstride=SO3_WIREFRAME_RSTRIDE, cstride=SO3_WIREFRAME_CSTRIDE,
                              color='k', linewidth=0.22, alpha=0.45)
        
        # NOTE: removed large semi-transparent coordinate planes per style request.
        # The visualization now relies on the sphere wireframe backdrop and thin axes only.
        
        # draw rotated axes (thinner lines, kept for reference)
        axis_len = 1.5
        axes_colors = ['red', 'green', 'blue']
        axes_labels = ['X', 'Y', 'Z']
        base_axes = np.eye(3) * axis_len
        rotated_axes = base_axes @ R.T

        for j in range(3):
            ax.quiver(0, 0, 0,
                     rotated_axes[j, 0], rotated_axes[j, 1], rotated_axes[j, 2],
                     color=axes_colors[j], arrow_length_ratio=0.06, linewidth=SO3_AXIS_LINEWIDTH)
            ax.text(rotated_axes[j, 0] * 1.08,
                   rotated_axes[j, 1] * 1.08,
                   rotated_axes[j, 2] * 1.08,
                   axes_labels[j], color=axes_colors[j], fontsize=8, fontweight='bold')
        
        ax.set_xlim([-1.6, 1.6])
        ax.set_ylim([-1.6, 1.6])
        ax.set_zlim([-1.6, 1.6])
        ax.set_title(result['name'], fontsize=11, fontweight='bold', pad=5)
        
        # clean background
        # remove pane backgrounds but keep axes lines/quivers
        try:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')
        except Exception:
            pass
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1, 1, 1])
    
    # colorbar on far right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=-vabs, vmax=vabs))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Field Value', fontsize=10)
    
    plt.subplots_adjust(left=0.02, right=0.90, top=0.95, bottom=0.05, wspace=0.05, hspace=0.1)
    save_figure(fig, 'so3_rotation_equivariance', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def run_rotation_equivariance_experiments(verbose=True):
    # run all rotation experiments
    set_seed(SEED)
    
    so2_results = run_so2_equivariance(seed=SEED)
    create_so2_figure(so2_results, verbose=verbose)
    
    if verbose:
        print(f"\n  so(2) complete")
    
    so3_results = run_so3_equivariance(seed=SEED)
    create_so3_figure(so3_results, verbose=verbose)
    
    if verbose:
        print(f"  so(3) complete")
    
    return so2_results, so3_results
