import numpy as np
from ase.io import read
from phonon_unfolding.displacement import get_displacement_atoms
import argparse

def generate_phonopy_yaml(input_file, supercell_dim):
    """
    Generate Phonopy YAML file with displacement information.

    :param input_file: Path to the input file containing the relaxed atomic structure.
    :param supercell_dim: List of three integers representing the supercell dimensions.
    :return: 0 upon successful completion.
    """
    # Load the relaxed atomic structure
    relaxed_atoms = read(input_file)

    # Generate displacement atoms and initialize the Phonopy object
    phonon, ase_disp_list = get_displacement_atoms(relaxed_atoms, supercell_dim)

    # Save the Phonopy object with displacements to a YAML file
    phonon.save('phonon_with_displacements.yaml')

    # Print generated displacements
    for i, d in enumerate(phonon.displacements):
        print(f"Displacement {i}: {d}")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Phonopy YAML file with displacement information")
    parser.add_argument('input_file', type=str, help='Input file containing the relaxed atomic structure')
    parser.add_argument('supercell_dim', type=int, nargs=3, help='Supercell dimensions as three integers')

    args = parser.parse_args()

    generate_phonopy_yaml(args.input_file, args.supercell_dim)
