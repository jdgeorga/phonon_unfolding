#!/usr/bin/env python

import h5py
import numpy as np
from phonopy import load
from phonopy.interface.qe import read_pwscf
from phonopy.structure.cells import get_supercell
import sys

def read_fc2_from_hdf5(filename):
    """Read force constants from phonopy's HDF5 file."""
    with h5py.File(filename, 'r') as f:
        fc = f['force_constants'][...]
        p2s_map = f['p2s_map'][...] if 'p2s_map' in f else None
    return fc, p2s_map

def write_q2r_fc(filename, fc, phonon, dimension):
    """Write force constants in q2r.x format.
    
    Parameters:
    -----------
    filename : str
        Output filename
    fc : ndarray
        Force constants array from phonopy
    phonon : Phonopy
        Phonopy object containing cell information
    dimension : array_like
        Supercell dimensions
    """
    primitive = phonon.primitive
    supercell = phonon.supercell
    natom = len(primitive)
    ndim = np.prod(dimension)
    
    with open(filename, 'w') as f:
        # Write header
        f.write(f"  {1}  {natom}  {0}  {0.0}  {0.0}  {0.0}\n")  # ntyp, nat, ibrav
        f.write("  0.0000000  0.0000000  0.0000000\n" * 3)  # Lattice (not used)
        f.write("  X  0.0  0.0\n" * natom)  # Atomic positions (not used)
        f.write("F\n")  # No dielectric/born data
        f.write(f" {dimension[0]}  {dimension[1]}  {dimension[2]}\n")  # Supercell dimensions
        
        # Write force constants
        # Convert from eV/Å² to Ry/au²
        conversion = 0.5291772106712**2 / 13.605693122994
        
        for k in range(3):
            for l in range(3):
                for i in range(natom):
                    for j in range(natom):
                        f.write(f"    {i+1}    {j+1}    {k+1}    {l+1}\n")
                        for n in range(ndim):
                            fc_val = fc[j, i*ndim + n, l, k] * conversion
                            f.write(f"  0  0  0  {fc_val:18.10f}\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: phonopy_to_qe.py phonopy.yaml fc2.hdf5")
        sys.exit(1)
        
    yaml_filename = sys.argv[1]
    fc_filename = sys.argv[2]
    
    # Read phonopy.yaml
    phonon = load(yaml_filename)
    
    # Read force constants
    fc, p2s_map = read_fc2_from_hdf5(fc_filename)
    
    # Determine supercell dimensions from phonon object
    dimension = phonon.supercell_matrix.diagonal()
    if not np.all(phonon.supercell_matrix == np.diag(dimension)):
        raise RuntimeError("Only diagonal supercell matrix is supported.")
    
    # Write force constants in q2r format
    write_q2r_fc('qe_force_constants', fc, phonon, dimension)
    
if __name__ == '__main__':
    main() 