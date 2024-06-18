import numpy as np
from ase.io import read
from phonopy import load
from phonon_unfolding.kpoints import (reciprocal_lattice_vectors,
                                      interpolate_qpoints,
                                      get_high_symmetry_qpoints)
from phonon_unfolding.plotting import (plot_high_sym_paths,
                                       plot_unfolded_bandstructure,
                                       plot_folded_bandstructure_with_projections)
from phonon_unfolding.unfolding import (get_Q_G_b_list_and_cart_Q_list,
                                        compute_coefficients,
                                        compute_qs_cart)
import argparse


def analyze_structure(input_file, disp_forces_file, phonopy_yaml_file, supercell_dim, atom_layers):
    """
    Analyze the structure to produce and plot the unfolded phonon band structure.

    :param input_file: Path to the input file containing the relaxed atomic structure.
    :param disp_forces_file: Path to the input file containing the displacement forces (numpy format).
    :param phonopy_yaml_file: Path to the input file containing the phonopy YAML data.
    :param supercell_dim: List of three integers representing the supercell dimensions.
    :param atom_layers: List of integers indicating the layer assignment of each atom.
    :return: 0 upon successful completion.
    """
    # Load the relaxed atomic structure
    relaxed_atoms = read(input_file)
    
    # Load the displacement forces
    disp_forces = np.load(disp_forces_file)
    
    # Load the phonopy YAML data
    phonon = load(phonopy_yaml_file)
    
    # Assign the displacement forces to the phonon object and compute force constants
    phonon.forces = disp_forces
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()

    # Plot the folded band structure with projections 
    plot_folded_bandstructure_with_projections(phonon,
                                               relaxed_atoms,
                                               atom_layers
                                               )

    # Adjust the primitive lattice vectors based on supercell scaling
    supercell_scale = (np.linalg.norm(phonon.primitive.cell, axis=-1) / 
                       np.linalg.norm(np.array(relaxed_atoms.cell), axis=-1))
    supercell_scale = np.around(supercell_scale)
    primitive_lattice_vectors = (phonon.primitive.cell.T / supercell_scale).T
    relaxed_atoms.set_cell(primitive_lattice_vectors, scale_atoms=True)

    # Adjust the supercell lattice vectors
    supercell_lattice_vectors = phonon.primitive.cell

    # Get high-symmetry q-points and labels from the relaxed atoms
    high_symmetry_q_points, high_symmetry_labels = get_high_symmetry_qpoints(relaxed_atoms)

    # Interpolate q-points along the high-symmetry path
    num_points = 100
    path_qpoints = interpolate_qpoints(high_symmetry_q_points, num_points)

    # Calculate reciprocal lattice vectors for primitive and supercell
    primitive_reciprocal_vectors = reciprocal_lattice_vectors(primitive_lattice_vectors)
    supercell_reciprocal_vectors = reciprocal_lattice_vectors(supercell_lattice_vectors)

    # Generate Q_G_b list and Cartesian Q list using KDTree
    Q_G_b_list, cart_Q_list, Q_points = get_Q_G_b_list_and_cart_Q_list(
        path_qpoints,
        primitive_reciprocal_vectors,
        supercell_reciprocal_vectors
        )

    # Set the band structure in the phonon object with eigenvectors
    phonon.set_band_structure(Q_points, is_eigenvectors=True)

    # Plot the high-symmetry paths
    plot_high_sym_paths(primitive_reciprocal_vectors,
                        supercell_reciprocal_vectors,
                        cart_Q_list)

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
        qs_cart, frequencies,
        coefficients,
        len(cart_Q_list),
        frequencies.shape[2],
        high_symmetry_labels
        )

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze structure to produce unfolded band structure")
    parser.add_argument('input_file', type=str, help='Input file containing the relaxed atomic structure')
    parser.add_argument('disp_forces_file', type=str, help='Input file containing the displacement forces (numpy format)')
    parser.add_argument('phonopy_yaml_file', type=str, help='Input file containing the phonopy YAML data')
    parser.add_argument('supercell_dim', type=int, nargs=3, help='Supercell dimensions as three integers')
    parser.add_argument('atom_layers', type=str, help='Comma-separated list indicating the layer assignment of each atom')
    
    args = parser.parse_args()
    
    # Parse the atom_layers argument into a list of integers
    atom_layers = list(map(int, args.atom_layers.split(',')))
    
    analyze_structure(args.input_file, args.disp_forces_file, args.phonopy_yaml_file, args.supercell_dim, atom_layers)
