from ase.calculators.lammpslib import LAMMPSlib

from ase.calculators.calculator import (
    Calculator,
    CalculatorError,
    CalculatorSetupError,
    all_changes,
)

from ase.data import atomic_numbers, atomic_masses
# from moirecompare.utils import rotate_to_x_axis
import numpy as np

import re

def remove_trailing_numbers(input_string):
    """
    Remove any trailing numbers from a string.
    
    Parameters
    ----------
    input_string : str
        The string from which to remove trailing numbers.
        
    Returns
    -------
    str
        String with trailing numbers removed.
        
    Raises
    ------
    TypeError
        If input is not a string.
    """
    if isinstance(input_string, str):
        return re.sub(r'\d+$', '', input_string)
    else:
        raise TypeError(f"Expected string or bytes-like object, but got {type(input_string)}")



class MonolayerLammpsCalculator(LAMMPSlib):
    """
    ASE calculator for monolayer systems using LAMMPS.
    
    This calculator is designed for intralayer interactions in monolayer systems,
    such as transition metal dichalcogenides (TMDs) or black phosphorus (BP).
    
    Parameters
    ----------
    atoms : Atoms
        ASE atoms object representing the monolayer system.
    layer_symbols : list[str]
        List of symbols representing different atom types in the monolayer.
    system_type : str, optional
        Type of the monolayer system, currently supports 'TMD' or 'BP', default is 'TMD'.
    intra_potential : str, optional
        Path to the intralayer potential file, default is 'tmd.sw'.
        
    Attributes
    ----------
    system_type : str
        Type of the monolayer system.
    layer_symbols : list[str]
        List of symbols representing different atom types.
    original_masses : list
        List of atomic masses for the atoms in the monolayer.
    original_chemical_symbols : list
        Original chemical symbols of the atoms.
    atom_types : array
        Array of atom types extracted from atoms object.
    intra_potential : str
        Path to the intralayer potential file.
    relative_layer_types : array
        Array mapping atom types to their relative positions.
    """
    implemented_properties = ["energy", "energies", "forces"]

    def __init__(self, atoms, layer_symbols, system_type='TMD', intra_potential='tmd.sw', comm=None):
        self.system_type = system_type
        self.layer_symbols = layer_symbols
        self.original_masses = [atomic_masses[m] for m in [atomic_numbers[remove_trailing_numbers(c)] for c in self.layer_symbols]]
        self.original_chemical_symbols = atoms.get_chemical_symbols()
        self.atom_types = atoms.arrays['atom_types']
        self.intra_potential = intra_potential
        self.comm = comm
        
        # Get unique types and their indices in the sorted order
        unique_types, inverse = np.unique(self.atom_types, return_inverse=True)
        
        # Create a new array with numbers replaced by their size order
        self.relative_layer_types = inverse

        # checks if atom_types match the layer_symbols
        if len(unique_types) != len(self.layer_symbols):
            raise ValueError("More/fewer atom types in Monolayer than layer_symbols provided.")

        if self.system_type is None: 
            print("Specify type of bilayer. Options are 'TMD' or 'BP'")

        if self.system_type == 'TMD':
            # Define LAMMPS commands for the TMD system
            cmds = [
                "pair_style sw/mod",
                f"pair_coeff * * {self.intra_potential} {self.layer_symbols[0]} {self.layer_symbols[1]} {self.layer_symbols[2]}",
                "neighbor        2.0 bin",
                "neigh_modify every 1 delay 0 check yes"]
            
            # Define fixed atom types and masses for the simulation
            fixed_atom_types = {'H': 1, 'He': 2, 'Li': 3}
            fixed_atom_type_masses = {'H': self.original_masses[0],
                                      'He': self.original_masses[1],
                                      'Li': self.original_masses[2]}
            
            # Set up the LAMMPS calculator with the specified commands
            LAMMPSlib.__init__(self,
                               lmpcmds=cmds,
                               atom_types=fixed_atom_types,
                               atom_type_masses=fixed_atom_type_masses,
                               keep_alive=True,
                               comm=self.comm)
            
        elif self.system_type == 'BP':
            # Define LAMMPS commands for the BP system
            cmds = [
                "pair_style sw/mod maxdelcs 0.25 0.35",
                f"pair_coeff * * {self.intra_potential} T T B B",
                "neighbor        2.0 bin",
                "neigh_modify every 1 delay 0 check yes"]
            
            # Define fixed atom types and masses for the simulation
            fixed_atom_types = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4}
            fixed_atom_type_masses = {'H': self.original_masses[0],
                                        'He': self.original_masses[1],
                                        'Li': self.original_masses[2],
                                        'Be': self.original_masses[3]}
            
            # Set up the LAMMPS calculator with the specified commands
            LAMMPSlib.__init__(self,
                                 lmpcmds=cmds,
                                 atom_types=fixed_atom_types,
                                 atom_type_masses=fixed_atom_type_masses,
                                 keep_alive=True,
                                 comm=self.comm)
    
    def calculate(self, 
                  atoms, 
                  properties=None,
                  system_changes=all_changes):
        """
        Perform the calculation using LAMMPS.
        
        Parameters
        ----------
        atoms : Atoms
            ASE atoms object to calculate properties for.
        properties : list, optional
            List of properties to calculate. If None, uses implemented_properties.
        system_changes : list, optional
            List of changes that have been made to the system since last calculation.
        """
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)
        
        if self.system_type is None: 
            print("Specify type of bilayer. Options are 'TMD' or 'BP'")

        if self.system_type == 'TMD':
            # Set atomic numbers based on relative layer types and perform calculation
            atoms.numbers = self.relative_layer_types + 1   
            self.propagate(atoms, properties, system_changes, 0)
            atoms.set_chemical_symbols(self.original_chemical_symbols)

        elif self.system_type == 'BP':
            # Set atomic numbers based on relative layer types and perform calculation
            atoms.numbers = self.relative_layer_types + 1
            self.propagate(atoms, properties, system_changes, 0)
            atoms.set_chemical_symbols(self.original_chemical_symbols)

    def minimize(self, atoms, method="fire"):
        """
        Perform energy minimization using LAMMPS.
        
        Parameters
        ----------
        atoms : Atoms
            ASE atoms object to minimize.
        method : str, optional
            Minimization algorithm to use, default is "fire".
        """
        # Define commands for minimization
        min_cmds = [f"min_style       {method}",
                    f"minimize        1.0e-6 1.0e-8 10000 10000",
                    "write_data      lammps.dat_min"]

        self.set(amendments=min_cmds)
        atoms.numbers = atoms.arrays["atom_types"] + 1
        self.propagate(atoms,
                       properties=["energy",
                                   'forces',
                                   'stress'],
                       system_changes=["positions",'pbc'],
                       n_steps=1)
        atoms.set_chemical_symbols(self.original_chemical_symbols)
        
    def clean_atoms(self, atoms):
        """
        Restore original atomic symbols and clean up LAMMPS instance.
        
        Parameters
        ----------
        atoms : Atoms
            ASE atoms object to clean up.
        """
        atoms.set_chemical_symbols(self.original_chemical_symbols)
        if self.started:
            self.lmp.close()
            self.started = False
            self.initialized = False
            self.lmp = None
         
class InterlayerLammpsCalculator(LAMMPSlib):
    """
    ASE calculator for interlayer interactions in layered systems using LAMMPS.
    
    This calculator is designed for interlayer interactions between stacked 2D materials,
    such as bilayer transition metal dichalcogenides (TMDs) or black phosphorus (BP).
    
    Parameters
    ----------
    atoms : Atoms
        ASE atoms object representing the layered system.
    layer_symbols : list[list[str]]
        Nested list of symbols representing different atom types in each layer.
    system_type : str, optional
        Type of the layered system, currently supports 'TMD' or 'BP', default is 'TMD'.
    inter_potential : str, optional
        Path to the interlayer potential file, default is 'WS.KC'.
        
    Attributes
    ----------
    system_type : str
        Type of the layered system.
    layer_symbols : list[list[str]]
        Nested list of symbols representing different atom types in each layer.
    original_chemical_symbols : list
        Original chemical symbols of the atoms.
    original_masses : list[list]
        Nested list of atomic masses for the atoms in each layer.
    atom_types : array
        Array of atom types extracted from atoms object.
    interlayer_potential : str
        Path to the interlayer potential file.
    relative_layer_types : array
        Array mapping atom types to their relative positions.
    """
    implemented_properties = ["energy", "energies", "forces", "free_energy"]

    def __init__(self, atoms, layer_symbols, system_type='TMD', inter_potential='WS.KC', comm=None):
        self.system_type = system_type
        self.layer_symbols = layer_symbols
        self.original_chemical_symbols = atoms.get_chemical_symbols()
        self.original_masses = [[atomic_masses[atomic_numbers[remove_trailing_numbers(n)]] for n in t] for t in self.layer_symbols]
        self.atom_types = atoms.arrays['atom_types']
        self.interlayer_potential = inter_potential
        self.comm = comm
        # Get unique types and their indices in the sorted order
        unique_types, inverse = np.unique(self.atom_types, return_inverse=True)
        
        # Create a new array with numbers replaced by their size order
        self.relative_layer_types = inverse

        # checks if atom_types match the layer_symbols
        if len(unique_types) != len([a for b in layer_symbols for a in b]):
            raise ValueError("More/fewer atom types in Monolayer than layer_symbols provided.")

        if self.system_type is None: 
            print("Specify type of bilayer. Options are 'TMD' or 'graphene'")

        if self.system_type == 'TMD':
            # Define LAMMPS commands for the TMD system with interlayer interactions
            cmds = [
                "pair_style hybrid/overlay kolmogorov/crespi/z 6.0 kolmogorov/crespi/z 6.0 kolmogorov/crespi/z 6.0 kolmogorov/crespi/z 6.0 lj/cut 20.0",
                f"pair_coeff 1 6 kolmogorov/crespi/z 1 {self.interlayer_potential}  {self.layer_symbols[0][0]} NULL NULL NULL NULL  {self.layer_symbols[1][2]}",
                f"pair_coeff 2 4 kolmogorov/crespi/z 2 {self.interlayer_potential}  NULL  {self.layer_symbols[0][1]} NULL {self.layer_symbols[1][0]} NULL NULL",
                f"pair_coeff 2 6 kolmogorov/crespi/z 3 {self.interlayer_potential}  NULL  {self.layer_symbols[0][1]} NULL NULL NULL  {self.layer_symbols[1][2]}",
                f"pair_coeff 1 4 kolmogorov/crespi/z 4 {self.interlayer_potential}   {self.layer_symbols[0][0]} NULL NULL {self.layer_symbols[1][0]} NULL NULL",
                "pair_coeff * * lj/cut 0.0 3.0",
                "neighbor        2.0 bin",
                "neigh_modify every 1 delay 0 check yes"]

            # Define fixed atom types and masses for the simulation
            fixed_atom_types = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6}
            fixed_atom_type_masses = {'H': self.original_masses[0][0],
                                      'He': self.original_masses[0][1],
                                      'Li': self.original_masses[0][2],
                                      'Be': self.original_masses[1][0],
                                      'B': self.original_masses[1][1],
                                      'C': self.original_masses[1][2]} 
            
            # Set up the LAMMPS calculator with the specified commands
            LAMMPSlib.__init__(self,
                      lmpcmds=cmds,
                      atom_types=fixed_atom_types,
                      atom_type_masses=fixed_atom_type_masses,
                      keep_alive = True,
                      comm=self.comm)
 
        elif self.system_type == 'BP':
            # Define LAMMPS commands for the BP system with interlayer interactions
            cmds = [
                "pair_style lj/cut 21.0",
                "pair_coeff * * 0.0 3.695",
                "pair_coeff 1*4 5*8 0.0103 3.405",
                "neighbor        2.0 bin",
                "neigh_modify every 1 delay 0 check yes"]
            
            # Define fixed atom types and masses for the simulation
            fixed_atom_types = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8}
            fixed_atom_type_masses = {'H': self.original_masses[0][0],
                                      'He': self.original_masses[0][1],
                                      'Li': self.original_masses[0][2],
                                      'Be': self.original_masses[0][3],
                                      'B': self.original_masses[1][0],
                                      'C': self.original_masses[1][1],
                                      'N': self.original_masses[1][2],
                                      'O': self.original_masses[1][3]}
            
            # Set up the LAMMPS calculator with the specified commands
            LAMMPSlib.__init__(self,
                               lmpcmds=cmds,
                               atom_types=fixed_atom_types,
                               atom_type_masses=fixed_atom_type_masses,
                               keep_alive=True,
                               comm=self.comm)
        
    def calculate(self,
                  atoms,
                  properties=None,
                  system_changes=all_changes):
        """
        Perform the calculation using LAMMPS for interlayer interactions.
        
        Parameters
        ----------
        atoms : Atoms
            ASE atoms object to calculate properties for.
        properties : list, optional
            List of properties to calculate. If None, uses implemented_properties.
        system_changes : list, optional
            List of changes that have been made to the system since last calculation.
        """
        if properties is None:
            properties = self.implemented_properties
        
        Calculator.calculate(self, atoms, properties, system_changes)

        if self.system_type is None: 
            print("Specify type of bilayer. Options are 'TMD' or 'graphene'")

        # Set atomic numbers based on relative layer types and perform calculation
        atoms.numbers = self.relative_layer_types + 1
        self.propagate(atoms, properties, system_changes, 0)
        atoms.set_chemical_symbols(self.original_chemical_symbols)

