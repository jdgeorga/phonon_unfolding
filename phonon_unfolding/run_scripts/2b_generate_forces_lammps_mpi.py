from mpi4py import MPI
from ase.io import read
import numpy as np
from phonopy import load
from phonon_unfolding.utils import phonopy_atoms_to_ase
import time

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

def add_lammps_calculators_to_atoms(atoms,layer_symbols, comm=None):
    """
    Add calculators to atoms for force calculations.

    :param atoms: List of ASE Atoms objects with displacements.
    :param intralayer_model: Path to the intralayer model file.
    :param interlayer_model: Path to the interlayer model file.
    :param layer_symbols: List of layer symbols.
    """

    from lammps_calculator import (MonolayerLammpsCalculator, InterlayerLammpsCalculator)
    from n_layer import NLayerCalculator
    
    intralayer_calcs = [
        MonolayerLammpsCalculator(atoms[0][atoms[0].arrays['atom_types'] < 3],
                                  layer_symbols[0],
                                  system_type='TMD',
                                  intra_potential='tmd.sw',
                                  comm=comm)
    ]
    interlayer_calcs = []

    for i in np.arange(1, len(layer_symbols)):
        layer_atoms = atoms[0][np.logical_and(atoms[0].arrays['atom_types'] >= i * 3,
                                              atoms[0].arrays['atom_types'] < (i + 1) * 3)]
        intralayer_calcs.append(MonolayerLammpsCalculator(layer_atoms,
                                                          layer_symbols=layer_symbols[i],
                                                          system_type='TMD',
                                                          intra_potential = 'tmd.sw',
                                                          comm=comm))
        bilayer_atoms = atoms[0][np.logical_and(atoms[0].arrays['atom_types'] >= (i - 1) * 3,
                                                atoms[0].arrays['atom_types'] < (i + 1) * 3)]
        interlayer_calcs.append(InterlayerLammpsCalculator(bilayer_atoms,
                                                           layer_symbols=layer_symbols[i - 1:i + 1],
                                                           system_type='TMD',
                                                           comm=comm))

    n_layer_calc = NLayerCalculator(atoms, intralayer_calcs, interlayer_calcs, layer_symbols)
    # n_layer_calc = intralayer_calcs[0]


    # Assign calculators to atoms
    for a in atoms:
        a.calc = n_layer_calc


def main(input_file,
         yaml_file,
         layer_symbols,
         calculator_type,
         out_file,
         intralayer_MoS2_model=None,
         interlayer_MoS2_model=None):
    """
    Main function to compute and save forces for each displacement structure.

    :param input_file: Path to the input file containing the relaxed atomic structure.
    :param yaml_file: Path to the phonopy YAML file.
    :param layer_symbols: List of layer symbols.
    :param calculator_type: Type of calculator to use (allegro or lammps).
    :param out_file: Path to the output file to save the forces.
    :param intralayer_MoS2_model: Path to the intralayer model file.
    :param interlayer_MoS2_model: Path to the interlayer model file.
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
        
        print(f"Number of displacement configurations: {len(ase_disp_list)}")
        print(f"Number of atoms in each configuration: {len(ase_disp_list[0])}")

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

    # Initialize calculators
    init_calc_time = time.time()

    if calculator_type == "allegro":
        add_allegro_calculators_to_atoms(ase_sublist, intralayer_MoS2_model, interlayer_MoS2_model, layer_symbols)

    elif calculator_type == "lammps":
        add_lammps_calculators_to_atoms(ase_sublist, layer_symbols, comm=MPI.COMM_SELF)

    forces = [a.get_forces() for a in ase_sublist]

    calc_time = time.time()

    # Gather all forces at the root process
    all_forces = comm.gather(forces, root=0)

    gather_time = time.time()

    if rank == 0:
        # Flatten the list of lists
        all_forces = [item for sublist in all_forces for item in sublist]
        disp_forces = np.array(all_forces)
        np.save(out_file, disp_forces)
        end_time = time.time()
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"Time to scatter data: {scatter_time - start_time:.2f} seconds")
        print(f"Time to initialize calculators: {init_calc_time - scatter_time:.2f} seconds")
        print(f"Time for calculations: {calc_time - init_calc_time:.2f} seconds")
        print(f"Time to gather results: {gather_time - calc_time:.2f} seconds")

if __name__ == "__main__":

    input_file = "MoS2-WSe2_bilayer_lammps_relax.xyz" # Relaxed Supercell
    phonopy_yaml_file = "phonon_with_displacements.yaml" #phonopy_yaml from generate_phonopy_yaml.py

    # intralayer_MoS2_model = "mos2_intra_no_shift.pth"
    # interlayer_MoS2_model = "interlayer_beefy.pth"
    layer_symbols = [["Mo", "S", "S"], ["W", "Se", "Se"]]

    calculator_type = "lammps"

    out_file = "disp_forces.npy"

    main(input_file, phonopy_yaml_file, layer_symbols, calculator_type, out_file)

    # Finalize the MPI environment
    MPI.Finalize()
