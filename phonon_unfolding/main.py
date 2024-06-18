from phonon_unfolding.displacement import get_displacement_atoms
from phonon_unfolding.kpoints import reciprocal_lattice_vectors, interpolate_qpoints, get_high_symmetry_qpoints
from phonon_unfolding.plotting import plot_high_sym_paths, plot_unfolded_bandstructure
from phonon_unfolding.unfolding import get_Q_G_b_list_and_cart_Q_list, compute_coefficients, compute_qs_cart
import numpy as np
from ase.io import read
from phonopy import Phonopy

def main(relaxed_atoms, disp_forces, primitive_original_unit_cell):
    """
    Main function to compute and plot the unfolded phonon band structure.

    :param relaxed_atoms: ASE Atoms object of the relaxed structure.
    :param disp_forces: Array of displacement forces.
    :param primitive_original_unit_cell: ASE Atoms object of the original primitive unit cell.
    :return: 0 upon successful completion.
    """
    # Generate displacement atoms and initialize the Phonopy object
    phonon, ase_disp_list = get_displacement_atoms(relaxed_atoms, [3, 3, 1])

    # Assign the displacement forces to the phonon object and compute force constants
    phonon.forces = disp_forces
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()

    # Adjust the primitive lattice vectors based on supercell scaling
    supercell_scale = (np.linalg.norm(phonon.primitive.cell, axis=-1) / np.linalg.norm(np.array(primitive_original_unit_cell.cell), axis=-1))
    supercell_scale = np.around(supercell_scale)
    primitive_lattice_vectors = (phonon.primitive.cell.T / supercell_scale).T

    # Set the cell for the original primitive unit cell
    primitive_original_unit_cell.set_cell(primitive_lattice_vectors, scale_atoms=True)

    # Adjust the supercell lattice vectors
    supercell_lattice_vectors = phonon.primitive.cell

    # Get high-symmetry q-points and labels from the original primitive unit cell
    high_symmetry_q_points, high_symmetry_labels = get_high_symmetry_qpoints(primitive_original_unit_cell)

    # Interpolate q-points along the high-symmetry path
    num_points = 100
    path_qpoints = interpolate_qpoints(high_symmetry_q_points, num_points)

    # Calculate reciprocal lattice vectors for primitive and supercell
    primitive_reciprocal_vectors = reciprocal_lattice_vectors(primitive_lattice_vectors)
    supercell_reciprocal_vectors = reciprocal_lattice_vectors(supercell_lattice_vectors)

    # Generate Q_G_b list and Cartesian Q list using KDTree
    Q_G_b_list, cart_Q_list, Q_points = get_Q_G_b_list_and_cart_Q_list(
        path_qpoints, primitive_reciprocal_vectors, supercell_reciprocal_vectors)

    # Set the band structure in the phonon object with eigenvectors
    phonon.set_band_structure(Q_points, is_eigenvectors=True)

    # Plot the high-symmetry paths
    plot_high_sym_paths(primitive_reciprocal_vectors, supercell_reciprocal_vectors, cart_Q_list)

    # Extract frequencies and eigenvectors from the phonon band structure
    phonon_band_structure = phonon.get_band_structure_dict()
    frequencies = np.array(phonon_band_structure['frequencies'])
    eigenvectors = np.array(phonon_band_structure['eigenvectors'])

    # Compute the coefficients and Cartesian q-points for the unfolded band structure
    coefficients = compute_coefficients(
        frequencies, eigenvectors, cart_Q_list, primitive_reciprocal_vectors, phonon)
    qs_cart = compute_qs_cart(cart_Q_list)

    # Plot the unfolded band structure
    plot_unfolded_bandstructure(
        qs_cart, frequencies, coefficients, len(cart_Q_list), frequencies.shape[2], high_symmetry_labels)

    return 0

if __name__ == "__main__":
    print("Executing Phonon Unfolding module")
    atoms = read("relaxed_MoS2_1D_13.89deg_8atoms.xyz", format="extxyz", index = -1)
    disp_forces = np.load("disp_forces_8.npy")
    import pymoire as pm
    materials_db_path = pm.materials.get_materials_db_path()
    layer_1 = pm.read_monolayer(materials_db_path / 'MoS2.cif')
    layer_2 = pm.read_monolayer(materials_db_path / 'MoS2.cif')
    layer_1.positions -= layer_1.positions[0]
    layer_2.positions -= layer_2.positions[0]
    layer_1.arrays['atom_types'] = np.array([0, 2, 1], dtype=int)
    layer_2.arrays['atom_types'] = np.array([0, 2, 1], dtype=int)
    primitive_original_unit_cell = layer_1 + layer_2
    main(atoms, disp_forces, primitive_original_unit_cell)
else:
    print("Importing phonon_unfolding module")
