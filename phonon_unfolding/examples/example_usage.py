
from phonon_unfolding.main import main
from ase.io import read
import numpy as np

# Load your structure and displacement forces
atoms = read("path/to/your/structure.xyz")
disp_forces = np.load("path/to/your/disp_forces.npy")

# Define primitive unit cell
import pymoire as pm
materials_db_path = pm.materials.get_materials_db_path()
layer_1 = pm.read_monolayer(materials_db_path / 'MoS2.cif')
layer_2 = pm.read_monolayer(materials_db_path / 'MoS2.cif')
layer_1.positions -= layer_1.positions[0]
layer_2.positions -= layer_2.positions[0]
layer_1.arrays['atom_types'] = np.array([0, 2, 1], dtype=int)
layer_2.arrays['atom_types'] = np.array([0, 2, 1], dtype=int)
primitive_unit_cell = layer_1 + layer_2

# Call the main function
main(atoms, disp_forces, primitive_unit_cell)
