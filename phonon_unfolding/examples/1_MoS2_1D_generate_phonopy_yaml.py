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


    input_file = "relaxed_MoS2_1D_3.4deg_30atoms.xyz" # Relaxed Supercell
    supercell_dim = [1, 3, 1]

    generate_phonopy_yaml(input_file, supercell_dim)
