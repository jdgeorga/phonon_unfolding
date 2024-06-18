import numpy as np
from ase.io import read
from moirecompare.calculators import AllegroCalculator, NLayerCalculator
from mpi4py import MPI
import argparse
import time
from phonon_unfolding.displacement import get_displacement_atoms
import warnings
warnings.filterwarnings("ignore")

def process_displacement(a, intralayer_model, interlayer_model, layer_symbols):
    intralayer_calcs = [AllegroCalculator(a[a.arrays['atom_types'] < 3], layer_symbols[0], model_file=intralayer_model)]
    interlayer_calcs = []

    for i in np.arange(1, len(layer_symbols)):
        layer_atoms = a[np.logical_and(a.arrays['atom_types'] >= i * 3, a.arrays['atom_types'] < (i + 1) * 3)]
        intralayer_calcs.append(AllegroCalculator(layer_atoms, layer_symbols=layer_symbols[i], model_file=intralayer_model, device='cpu'))
        bilayer_atoms = a[np.logical_and(a.arrays['atom_types'] >= (i - 1) * 3, a.arrays['atom_types'] < (i + 1) * 3)]
        interlayer_calcs.append(AllegroCalculator(bilayer_atoms, layer_symbols=layer_symbols[i - 1:i + 1], model_file=interlayer_model, device='cpu'))

    n_layer_calc = NLayerCalculator(a, intralayer_calcs, interlayer_calcs, layer_symbols)
    a.calc = n_layer_calc
    a.calc.calculate(a)
    return a.get_forces()

def generate_forces(input_file, output_file, supercell_dim, intralayer_model, interlayer_model):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    atoms = read(input_file)
    phonon, ase_disp_list = get_displacement_atoms(atoms, supercell_dim)

    if rank == 0:
        ase_disp_list = ase_disp_list[:]
        for a in ase_disp_list:
            a.arrays['atom_types'] = atoms.arrays['atom_types'].repeat(3)
    else:
        ase_disp_list = None

    scatter_time = time.time()

    chunks = comm.scatter([ase_disp_list[i::size] for i in range(size)], root=0)
    print(f"Process {rank} received chunk: {len(chunks)}")

    layer_symbols = [["Mo", "S", "S"], ["Mo", "S", "S"]]

    forces = [process_displacement(a, intralayer_model, interlayer_model, layer_symbols) for a in chunks]

    all_forces = comm.gather(forces, root=0)

    if rank == 0:
        all_forces = [item for sublist in all_forces for item in sublist]
        disp_forces = np.array(all_forces)
        np.save(output_file, disp_forces)

    gather_time = time.time()

    if rank == 0:
        print(f"Total time: {gather_time - scatter_time:.2f} seconds")

    MPI.Finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate forces on a particular structure")
    parser.add_argument('input_file', type=str, help='Input file containing the atomic structure')
    parser.add_argument('output_file', type=str, help='Output file for the displacement forces (numpy format)')
    parser.add_argument('supercell_dim', type=int, nargs=3, help='Supercell dimensions as three integers')
    parser.add_argument('intralayer_model', type=str, help='Path to the intralayer model file')
    parser.add_argument('interlayer_model', type=str, help='Path to the interlayer model file')
    
    args = parser.parse_args()
    generate_forces(args.input_file, args.output_file, args.supercell_dim, args.intralayer_model, args.interlayer_model)
