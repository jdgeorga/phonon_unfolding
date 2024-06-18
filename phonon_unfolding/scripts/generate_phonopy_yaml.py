import numpy as np
from ase.io import read
from phonon_unfolding.displacement import get_displacement_atoms
from phonon_unfolding.kpoints import get_high_symmetry_qpoints, reciprocal_lattice_vectors, interpolate_qpoints
from phonon_unfolding.plotting import plot_high_sym_paths
import argparse

def generate_phonopy_yaml(input_file, supercell_dim, output_file):
    relaxed_atoms = read(input_file)
    phonon, ase_disp_list = get_displacement_atoms(relaxed_atoms, supercell_dim)
    
    # Save phonopy YAML file
    phonon.save(output_file)
    
    # Plot high symmetry paths
    primitive_lattice_vectors = phonon.primitive.cell
    primitive_reciprocal_vectors = reciprocal_lattice_vectors(primitive_lattice_vectors)
    high_symmetry_q_points, high_symmetry_labels = get_high_symmetry_qpoints(relaxed_atoms)
    num_points = 100
    path_qpoints = interpolate_qpoints(high_symmetry_q_points, num_points)
    
    plot_high_sym_paths(primitive_reciprocal_vectors, reciprocal_lattice_vectors(phonon.primitive.cell), path_qpoints)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Phonopy YAML file and plot high symmetry paths")
    parser.add_argument('input_file', type=str, help='Input file containing the relaxed atomic structure')
    parser.add_argument('supercell_dim', type=int, nargs=3, help='Supercell dimensions as three integers')
    parser.add_argument('output_file', type=str, help='Output YAML file')
    
    args = parser.parse_args()
    generate_phonopy_yaml(args.input_file, args.supercell_dim, args.output_file)
