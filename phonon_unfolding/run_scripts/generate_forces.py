from mpi4py import MPI
from ase.io import read
import numpy as np
from phonopy import load
from moirecompare.utils import phonopy_atoms_to_ase
import time

from moirecompare.calculators import AllegroCalculator, NLayerCalculator
import warnings
warnings.filterwarnings("ignore")

def get_displacement_atoms_from_yaml(yaml_file):
    """
    Get displacement atoms from a phonopy YAML file.

    :param yaml_file: Path to the phonopy YAML file.
    :return: Phonopy object and list of ASE atoms with displacements.
    """
    phonon = load(yaml_file)
    phonon_displacement_list = phonon.get_supercells_with_displacements()
    ase_disp_list = [phonopy_atoms_to_ase(x) for x in phonon_displacement_list]
    return phonon, ase_disp_list

def add_calculators_to_atoms(atoms, intralayer_model, interlayer_model, layer_symbols):
    """
    Add calculators to atoms for force calculations.

    :param atoms: List of ASE Atoms objects with displacements.
    :param intralayer_model: Path to the intralayer model file.
    :param interlayer_model: Path to the interlayer model file.
    :param layer_symbols: List of layer symbols.
    """
    intralayer_calcs = [
        AllegroCalculator(atoms[0][atoms[0].arrays['atom_types'] < 3], layer_symbols[0],
                          model_file=intralayer_model, device='cpu')
    ]
    interlayer_calcs = []

    for i in np.arange(1, len(layer_symbols)):
        layer_atoms = atoms[0][np.logical_and(atoms[0].arrays['atom_types'] >= i * 3,
                                              atoms[0].arrays['atom_types'] < (i + 1) * 3)]
        intralayer_calcs.append(AllegroCalculator(layer_atoms, layer_symbols=layer_symbols[i],
                                                  model_file=intralayer_model, device='cpu'))
        bilayer_atoms = atoms[0][np.logical_and(atoms[0].arrays['atom_types'] >= (i - 1) * 3,
                                                atoms[0].arrays['atom_types'] < (i + 1) * 3)]
        interlayer_calcs.append(AllegroCalculator(bilayer_atoms, layer_symbols=layer_symbols[i - 1:i + 1],
                                                  model_file=interlayer_model, device='cpu'))

    n_layer_calc = NLayerCalculator(atoms, intralayer_calcs, interlayer_calcs, layer_symbols)

    # Assign calculators to atoms
    for a in atoms:
        a.calc = n_layer_calc

def main(input_file, yaml_file, intralayer_MoS2_model, interlayer_MoS2_model, layer_symbols):
    """
    Main function to compute and save forces for each displacement structure.

    :param input_file: Path to the input file containing the relaxed atomic structure.
    :param yaml_file: Path to the phonopy YAML file.
    :param intralayer_MoS2_model: Path to the intralayer model file.
    :param interlayer_MoS2_model: Path to the interlayer model file.
    :param layer_symbols: List of layer symbols.
    """
    # Initialize the MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start_time = time.time()

    relaxed_atoms = read(input_file, format="extxyz", index=-1)

    if rank == 0:
        phonon, ase_disp_list = get_displacement_atoms_from_yaml(yaml_file)
        supercell_repeats = len(ase_disp_list[0]) // len(relaxed_atoms)

        for a in ase_disp_list:
            a.arrays['atom_types'] = relaxed_atoms.arrays['atom_types'].repeat(supercell_repeats)

        # Calculate the chunk sizes for each process
        avg_chunk_size = len(ase_disp_list) // size
        remainder = len(ase_disp_list) % size
        chunks = []
        start = 0
        for i in range(size):
            end = start + avg_chunk_size + (1 if i < remainder else 0)
            chunks.append(ase_disp_list[start:end])
            start = end
    else:
        chunks = None

    scatter_time = time.time()

    # Scatter the list to all processes
    ase_sublist = comm.scatter(chunks, root=0)
    print(f"Process {rank} received chunk: {len(ase_sublist)}")

    # Initialize calculators
    init_calc_time = time.time()

    add_calculators_to_atoms(ase_sublist, intralayer_MoS2_model, interlayer_MoS2_model, layer_symbols)

    forces = [a.get_forces() for a in ase_sublist]

    calc_time = time.time()

    # Gather all forces at the root process
    all_forces = comm.gather(forces, root=0)

    gather_time = time.time()

    if rank == 0:
        # Flatten the list of lists
        all_forces = [item for sublist in all_forces for item in sublist]
        disp_forces = np.array(all_forces)
        np.save('disp_forces.npy', disp_forces)
        end_time = time.time()
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"Time to scatter data: {scatter_time - start_time:.2f} seconds")
        print(f"Time to initialize calculators: {init_calc_time - scatter_time:.2f} seconds")
        print(f"Time for calculations: {calc_time - init_calc_time:.2f} seconds")
        print(f"Time to gather results: {gather_time - calc_time:.2f} seconds")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate forces for displacement structures using MPI")
    parser.add_argument('input_file', type=str, help='Input file containing the relaxed atomic structure')
    parser.add_argument('yaml_file', type=str, help='Path to the phonopy YAML file with displacements')

    args = parser.parse_args()

    intralayer_MoS2_model = "mos2_intra_no_shift.pth"
    interlayer_MoS2_model = "interlayer_beefy.pth"
    layer_symbols = [["Mo", "S", "S"], ["Mo", "S", "S"]]

    main(args.input_file, args.yaml_file, intralayer_MoS2_model, interlayer_MoS2_model, layer_symbols)

    # Finalize the MPI environment
    MPI.Finalize()
