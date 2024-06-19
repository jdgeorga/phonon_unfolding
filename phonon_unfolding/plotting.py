import matplotlib.pyplot as plt
import numpy as np
from phonon_unfolding.kpoints import get_high_symmetry_qpoints, interpolate_qpoints
from phonon_unfolding.utils import proj_phonons_nlayer

def plot_high_sym_paths(primitive_reciprocal_vectors, supercell_reciprocal_vectors, cart_Q_list):
    """
    Plot the high-symmetry paths in the reciprocal lattice.

    :param primitive_reciprocal_vectors: 3x3 array-like of primitive reciprocal lattice vectors.
    :param supercell_reciprocal_vectors: 3x3 array-like of supercell reciprocal lattice vectors.
    :param cart_Q_list: List of Cartesian Q points.
    """
    from scipy.spatial import Voronoi, voronoi_plot_2d

    def plot_wigner_seitz_cells(lattice_vectors, plot_range=5, color='k', label="primitive cell", ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        # Generate lattice points
        points = []
        for i in range(-plot_range, plot_range + 1):
            for j in range(-plot_range, plot_range + 1):
                point = i * lattice_vectors[0][:2] + j * lattice_vectors[1][:2]
                points.append(point)
        points = np.array(points)

        # Compute Voronoi diagram
        vor = Voronoi(points)

        # Plot the Voronoi diagram
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors=color, line_width=2, line_alpha=0.6, point_size=2)
  
        # Plot lattice points
        ax.plot(points[:, 0], points[:, 1], 'o', color=color, label=f'{label} lattice points')

        return ax

    # Extract Cartesian q-points
    qs_cart = np.array([q_cart for segment in cart_Q_list for q_cart, Q_cart, G_b_cart in segment])

    # Flatten Q_G_b_list and extract Q points
    Q_points = np.array([Q for segment in cart_Q_list for _, Q, _ in segment])

    # Plot Wigner-Seitz cells for both the primitive and supercell
    fig, ax = plt.subplots(figsize=(10, 10))

    ax = plot_wigner_seitz_cells(primitive_reciprocal_vectors, plot_range=5, color='blue', label="primitive cell", ax=ax)
    ax = plot_wigner_seitz_cells(supercell_reciprocal_vectors, plot_range=15, color='red', label="supercell", ax=ax)

    # Plot the q-points and Q points
    ax.scatter(qs_cart[:, 0], qs_cart[:, 1], color='blue', label='high-symmetry q-points in primitive')
    ax.scatter(Q_points[:, 0], Q_points[:, 1], color='green', label='Folded high-symmetry Q_points in supercell')

    # Set plot properties
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title('Wigner-Seitz Cells: Primitive (Blue) and Supercell (Red)')
    plt.savefig("Path_on_Wigner_Seitz_cells.png", dpi=300, bbox_inches="tight")

def plot_unfolded_bandstructure(qs_cart, frequencies, coefficients, num_segments, num_bands, high_symmetry_labels):
    """
    Plot the unfolded phonon band structure.

    :param qs_cart: List of Cartesian q-points.
    :param frequencies: Array of phonon frequencies.
    :param coefficients: Array of unfolded coefficients.
    :param num_segments: Number of q-point segments.
    :param num_bands: Number of phonon bands.
    :param high_symmetry_labels: List of high-symmetry point labels.
    """
    plt.figure(figsize=(8, 6))

    qs_cart_dist = [np.linalg.norm(np.array(segment) - np.array(segment[0]), axis=-1) for segment in qs_cart]

    for i in range(len(qs_cart_dist) - 1):
        qs_cart_dist[i + 1] += qs_cart_dist[i][-1]
    
    cutoff_max = max([c.max() for c in coefficients])   
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
            end_idx = (i + 1) * decile_size if i < 29 else len(coeffs_flat)

            qs_decile = qs_flat[start_idx:end_idx]
            freqs_decile = freqs_flat[start_idx:end_idx]
            coeffs_decile = coeffs_flat[start_idx:end_idx]

            plt.scatter(qs_decile, freqs_decile * 33.356, c=coeffs_decile, cmap='Blues', s=1, alpha=1, vmin=0.0, vmax=cutoff_max, zorder=i + 1)

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
    plt.colorbar(label='Weight')

    plt.savefig('unfold_phonon_band_structure.png', dpi=300)

def plot_folded_bandstructure_with_projections(phonon, atoms, atom_layers):
    """
    Plot the folded phonon band structure with projections onto different layers.

    :param phonon: Phonopy object with the phonon band structure.
    :param atoms: ASE Atoms object of the structure.
    :param atom_layers: Array indicating the layer assignment of each atom.
    """
    # Get high-symmetry q-points and labels from the relaxed atoms
    high_symmetry_q_points, high_symmetry_labels = get_high_symmetry_qpoints(atoms)

    # Interpolate q-points along the high-symmetry path
    num_points = 100
    path_qpoints = interpolate_qpoints(high_symmetry_q_points, num_points)

    # Set the band structure in the phonon object with eigenvectors
    phonon.set_band_structure(path_qpoints, is_eigenvectors=True)

    # Extract distances and frequencies from the phonon band structure
    band_structure_dict = phonon.get_band_structure_dict()
    distances = band_structure_dict['distances']
    frequencies = band_structure_dict['frequencies']

    # Project phonons onto layers
    proj_layers = [proj_phonons_nlayer(atoms, phonon, j, atom_layers)[0] for j in band_structure_dict['eigenvectors']]

    plt.figure(figsize=(8, 6))

    num_segments = len(distances)
    for i_segment in range(num_segments):
        for i_band in range(phonon.unitcell.get_number_of_atoms() * 3):
            blended_colors = proj_layers[i_segment][:, i_band]

            band0_cond = blended_colors[:, 0] > 0.8
            band1_cond = blended_colors[:, 1] > 0.8
            band2_cond = blended_colors[:, 2] > 0.8
            hybrid_cond = np.logical_not(np.logical_or(np.logical_or(band0_cond, band1_cond), band2_cond))

            blended_colors = blended_colors / blended_colors.max(axis=-1)[:, None]

            plt.scatter(distances[i_segment][band0_cond],
                        frequencies[i_segment][band0_cond, i_band] * 33.356,
                        c=blended_colors[band0_cond], zorder=1, s=1)

            plt.scatter(distances[i_segment][band1_cond],
                        frequencies[i_segment][band1_cond, i_band] * 33.356,
                        c=blended_colors[band1_cond], zorder=2, s=1)

            plt.scatter(distances[i_segment][band2_cond],
                        frequencies[i_segment][band2_cond, i_band] * 33.356,
                        c=blended_colors[band2_cond], zorder=3, s=1)

            plt.scatter(distances[i_segment][hybrid_cond],
                        frequencies[i_segment][hybrid_cond, i_band] * 33.356,
                        c=blended_colors[hybrid_cond], zorder=4, s=1)

    if len(np.unique(atom_layers)) == 3:
        plt.plot(np.nan, np.nan, color=[1., 0., 0.], label='Layer 1', ls="-")
        plt.plot(np.nan, np.nan, color=[1., 1., 0.], label='Layer 1+2', ls="-")
        plt.plot(np.nan, np.nan, color=[0., 1., 0.], label='Layer 2', ls="-")
        plt.plot(np.nan, np.nan, color=[0., 1., 1.], label='Layer 2+3', ls="-")
        plt.plot(np.nan, np.nan, color=[0., 0., 1.], label='Layer 3', ls="-")
        plt.plot(np.nan, np.nan, color=[1., 0., 1.], label='Layer 1+3', ls="-")

    elif len(np.unique(atom_layers)) == 2:
        plt.plot(np.nan, np.nan, color=[1., 0., 0.], label='Layer 1', ls="-")
        plt.plot(np.nan, np.nan, color=[0., 0., 1.], label='Layer 2', ls="-")
        plt.plot(np.nan, np.nan, color=[1., 0., 1.], label='Layer 1+2', ls="-")

    elif len(np.unique(atom_layers)) == 1:
        plt.plot(np.nan, np.nan, color=[1., 0., 0.], label='Layer 1', ls="-")

    tick_labels = high_symmetry_labels
    tick_positions = [0] + [d[-1] for d in distances]

    for pos in tick_positions[1:]:
        plt.axvline(x=pos, c='grey')

    plt.xticks(tick_positions, tick_labels)
    plt.ylabel('Frequency (cm$^{-1}$)')
    plt.title("Folded Moire Phonon Band Structure")
    plt.axhline(0, c='black', lw=1, zorder=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig('phonon_band_structure_with_projections.png', dpi=300)