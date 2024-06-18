# Functions for plotting
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np

def plot_wigner_seitz_cells(lattice_vectors, plot_range=5, color='k', label="primitive cell", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    points = []
    for i in range(-plot_range, plot_range + 1):
        for j in range(-plot_range, plot_range + 1):
            point = i * lattice_vectors[0][:2] + j * lattice_vectors[1][:2]
            points.append(point)
    points = np.array(points)
    vor = Voronoi(points)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors=color, line_width=2, line_alpha=0.6, point_size=2)
    ax.plot(points[:, 0], points[:, 1], 'o', color=color, label=f'{label} lattice points')

    return ax

def plot_high_sym_paths(primitive_reciprocal_vectors, supercell_reciprocal_vectors, cart_Q_list):
    qs_cart = np.array([q_cart for segment in cart_Q_list for q_cart, Q_cart, G_b_cart in segment])
    Q_points = np.array([Q for segment in cart_Q_list for _, Q, _ in segment])
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = plot_wigner_seitz_cells(primitive_reciprocal_vectors, plot_range=5, color='blue', label="primitive cell", ax=ax)
    ax = plot_wigner_seitz_cells(supercell_reciprocal_vectors, plot_range=15, color='red', label="supercell", ax=ax)
    ax.scatter(qs_cart[:, 0], qs_cart[:, 1], color='blue', label='high-symmetry q-points in primitive')
    ax.scatter(Q_points[:, 0], Q_points[:, 1], color='green', label='Folded high-symmetry Q_points in supercell')
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title('Wigner-Seitz Cells: Primitive (Blue) and Supercell (Red)')
    plt.savefig("Path_on_Wigner_Seitz_cells.png", dpi=300, bbox_inches="tight")

def plot_unfolded_bandstructure(qs_cart, frequencies, coefficients, num_segments, num_bands, high_symmetry_labels):
    plt.figure(figsize=(8, 6))
    qs_cart_dist = [np.linalg.norm(np.array(segment) - np.array(segment[0]), axis=-1) for segment in qs_cart]
    for i in range(len(qs_cart_dist)-1):
        qs_cart_dist[i+1] += qs_cart_dist[i][-1]
    for i_segment in range(num_segments):
        coeffs_flat = coefficients[i_segment].flatten()
        qs_flat = np.repeat(qs_cart_dist[i_segment], num_bands)
        freqs_flat = frequencies[i_segment].flatten()
        cutoff_cond = coeffs_flat > 0.025
        coeffs_flat = coeffs_flat[cutoff_cond]
        qs_flat = qs_flat[cutoff_cond]
        freqs_flat = freqs_flat[cutoff_cond]
        sorted_indices = np.argsort(coeffs_flat)
        coeffs_flat = coeffs_flat[sorted_indices]
        qs_flat = qs_flat[sorted_indices]
        freqs_flat = freqs_flat[sorted_indices]
        decile_size = len(coeffs_flat) // 30
        for i in range(30):
            start_idx = i * decile_size
            end_idx = (i + 1) * decile_size if i < (30-1) else len(coeffs_flat)
            qs_decile = qs_flat[start_idx:end_idx]
            freqs_decile = freqs_flat[start_idx:end_idx]
            coeffs_decile = coeffs_flat[start_idx:end_idx]
            sc = plt.scatter(qs_decile, freqs_decile*33.356, c=coeffs_decile, cmap='Blues', s=1, alpha=1, vmin=0.0, vmax=0.15, zorder=i + 1)
    xticks = [x[-1] for x in qs_cart_dist]
    xticks.insert(0, 0)
    plt.xticks(xticks, high_symmetry_labels, rotation=45)
    plt.axvline(x=qs_cart_dist[0][0], color='grey', linestyle='--')
    for i in range(len(qs_cart_dist)):
        plt.axvline(x=qs_cart_dist[i][-1], color='grey', linestyle='--')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.ylim(-50, 600)
    plt.xlabel('Wave Vector')
    plt.ylabel('Frequency (cm$^{-1}$)')
    plt.title('Unfolded Phonon Band Structure')
    plt.colorbar(sc, label='Weight')
    plt.savefig(f'unfold_phonon_band_structure.png', dpi=300)
