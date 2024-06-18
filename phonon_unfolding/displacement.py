import numpy as np
from phonon_unfolding.utils import ase_to_phonopy_atoms, phonopy_atoms_to_ase
from phonopy import Phonopy

def get_displacement_atoms(ase_atom, supercell_n):
    """
    Generate displacement atoms for a given ASE Atoms object and supercell size.

    :param ase_atom: ASE Atoms object.
    :param supercell_n: List or array-like of three integers specifying the supercell dimensions.
    :return: Tuple containing the Phonopy object and a list of displaced ASE Atoms objects.
    """
    phonopy_atoms = ase_to_phonopy_atoms(ase_atom)
    phonon = Phonopy(phonopy_atoms, supercell_matrix=np.diag(supercell_n), log_level=2)
    phonon.generate_displacements(is_diagonal=True)
    phonon_displacement_list = phonon.get_supercells_with_displacements()
    ase_disp_list = [phonopy_atoms_to_ase(x) for x in phonon_displacement_list]
    return phonon, ase_disp_list


'''
# EXAMPLE USAGE


from phonon_unfolding.displacement import get_displacement_atoms
from ase.io import read

# Load your structure
atoms = read("path/to/your/structure.xyz")

# Define the supercell dimensions
supercell_n = [2, 3, 1]

# Generate displacement atoms
phonon, ase_disp_list = get_displacement_atoms(atoms, supercell_n)
'''
