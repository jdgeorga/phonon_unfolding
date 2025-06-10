from ase.io import read, write  # Import read and write functions from ASE for handling atomic structures
import numpy as np  # Import numpy for numerical operations
from n_layer import NLayerCalculator
from macewrapper import MACEWCalculator
from ase.optimize import FIRE, BFGS  # Import optimization algorithms from ASE

def mace_allegro_relax(input_file, output_file, mo_model_path, w_model_path, interlayer_model_path):
    """
    Function to relax a given atomic structure using MACE for intralayer and Allegro for interlayer interactions.
    
    Parameters:
    input_file (str): Path to the input structure file in extxyz format.
    output_file (str): Prefix for the output files to store relaxed structure and trajectory.
    mo_model_path (str): Path to the MACE model for MoS2.
    w_model_path (str): Path to the MACE model for WSe2.
    interlayer_model_path (str): Path to the Allegro model for interlayer interactions.
    """
    # Read the input atomic structure
    atoms = read(input_file, format="extxyz")


    # Define atomic symbols for each layer
    layer_symbols = [["Mo", "S", "S"],
                    ["W", "Se", "Se"]]

    # Set up intralayer calculators using MACE
    intralayer_calcs = [
        MACEWCalculator(atoms[atoms.arrays['atom_types'] < 3], layer_symbols[0],
                       model_file=mo_model_path, device='cpu', is_interlayer_calc=False)
    ]
    interlayer_calcs = []

    # Set up additional layers and interlayer interactions
    for i in np.arange(1, len(layer_symbols)):
        layer_atoms = atoms[np.logical_and(atoms.arrays['atom_types'] >= i * 3,
                                         atoms.arrays['atom_types'] < (i + 1) * 3)]
        intralayer_calcs.append(MACEWCalculator(layer_atoms, layer_symbols=layer_symbols[i],
                                              model_file=w_model_path, device='cpu'))
        bilayer_atoms = atoms[np.logical_and(atoms.arrays['atom_types'] >= (i - 1) * 3,
                                           atoms.arrays['atom_types'] < (i + 1) * 3)]
        interlayer_calcs.append(MACEWCalculator(bilayer_atoms, layer_symbols=layer_symbols[i - 1:i + 1],
                                                model_file=interlayer_model_path, is_interlayer_calc=True, device='cpu'))

    # Combine the calculators into an NLayerCalculator
    n_layer_calc = NLayerCalculator(atoms, intralayer_calcs, interlayer_calcs, layer_symbols)
    
    # Assign the combined calculator to the atoms object
    atoms.calc = n_layer_calc

    # Perform an initial calculation to get unrelaxed energy
    atoms.calc.calculate(atoms)
    print(f"Unrelaxed: Total_energy {atoms.calc.results['energy']:.3f} eV")

    # Set up the FIRE optimizer for structural relaxation
    dyn = FIRE(atoms, trajectory=f"{output_file}.traj", maxstep=0.05)
    dyn.run(fmax=1e-4, steps=1000)  # Run until the maximum force is below 8e-6 eV/Å

    # Print the relaxed energy
    print(f"Relaxed: Total_energy {atoms.calc.results['energy']:.3f} eV")

    # Import additional ASE modules for handling trajectories
    from ase.io.trajectory import Trajectory

    # Read the trajectory file generated during relaxation
    traj_path = f"{output_file}.traj"
    traj = Trajectory(traj_path)
    
    # Collect all images from the trajectory
    images = [atom for atom in traj]

    # Write all images to an output file in extxyz format
    write(f"{output_file}.traj.xyz", images, format="extxyz")
    
    write(f"{output_file}.xyz", atoms, format="extxyz")

if __name__ == "__main__":
    # Define the input XYZ file and output file prefix
    xyz_file_path = "MoS2-WSe2_bilayer_42a_25deg.xyz"
    out_file = 'MoS2_WSe2_mace_relax'

    # Define model paths
    mo_model_path = "MoS2-models_rmax2_stagetwo_compiled.model"
    w_model_path = "WSe2-models_rmax2_stagetwo_compiled.model"
    interlayer_model_path = "MoS2-WSe2-inter-for-jdg.model"

    # Call the relaxation function with specified input and output paths
    mace_allegro_relax(xyz_file_path, out_file, mo_model_path, w_model_path, interlayer_model_path) 