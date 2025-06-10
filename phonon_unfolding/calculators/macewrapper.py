from ase.calculators.calculator import (
    Calculator,
    CalculatorError,
    CalculatorSetupError,
    all_changes,
)
from ase.data import atomic_masses, atomic_numbers, chemical_symbols
from typing import List, Dict, Union
from pathlib import Path
from ase.atoms import Atoms
import numpy as np
import torch
from mace.calculators import MACECalculator

class MACEWCalculator(Calculator):
    # Define the properties that the calculator can handle
    implemented_properties = ["energy", "energies", "forces", "free_energy"]

    def __init__(self,
                 atoms: Atoms,
                 layer_symbols: list[str],
                 model_file: str,
                 device='cpu',
                 default_dtype='float32',
                 is_interlayer_calc=False,
                 **kwargs):
        """
        Initializes the MACEWCalculator with a given set of atoms, layer symbols, model file, and device.

        :param atoms: ASE atoms object.
        :param layer_symbols: List of symbols representing different layers in the structure.
        :param model_file: Path to the file containing the trained model.
        :param device: Device to run the calculations on, default is 'cpu'.
        :param default_dtype: Default data type for calculations ('float32' or 'float64').
        :param kwargs: Additional keyword arguments for the base class.
        """
        self.atoms = atoms  # ASE atoms object
        self.atom_types = atoms.arrays['atom_types']  # Extract atom types from atoms object
        self.device = device  # Device for computations
        self.layer_ids = atoms.arrays['layer_ids']
        self.is_interlayer_calc = is_interlayer_calc
        # Flatten the layer symbols list
        self.layer_symbols = [symbol for sublist in layer_symbols for symbol in (sublist if isinstance(sublist, list) else [sublist])]

        # Determine unique atom types and their indices
        unique_types, inverse = np.unique(self.atom_types, return_inverse=True)

        # Map atom types to their relative positions in the unique_types array
        self.relative_layer_types = inverse

        # Ensure the number of unique atom types matches the number of layer symbols provided
        if len(unique_types) != len(self.layer_symbols):
            raise ValueError("Mismatch between the number of atom types and provided layer symbols.")

        # Initialize the MACE calculator
        self.mace_calc = MACECalculator(
            model_paths=model_file,
            device=device,
            default_dtype=default_dtype,
            is_interlayer_calc=is_interlayer_calc,
            **kwargs
        )

        # Initialize the base Calculator class with any additional keyword arguments
        Calculator.__init__(self, **kwargs)

    def calculate(self,
                 atoms: Atoms = None,
                 properties=None,
                 system_changes=all_changes):
        """
        Performs the calculation for the given atoms and properties.

        :param atoms: ASE atoms object to calculate properties for.
        :param properties: List of properties to calculate. If None, uses implemented_properties.
        :param system_changes: List of changes that have been made to the system since last calculation.
        """
        # Default to implemented properties if none are specified
        if properties is None:
            properties = self.implemented_properties

        # Create a temporary copy of the atoms object
        tmp_atoms = atoms.copy()[:]
        tmp_atoms.calc = None  # Remove any attached calculator

        # Backup original atomic numbers and set new atomic numbers based on relative layer types
        original_atom_numbers = tmp_atoms.numbers.copy()
        # tmp_atoms.set_atomic_numbers(self.relative_layer_types + 1)
        tmp_atoms.arrays['atom_types'] = self.relative_layer_types
        tmp_atoms.arrays['layer_ids'] = self.layer_ids

        # Set the MACE calculator for the temporary atoms
        tmp_atoms.calc = self.mace_calc

        # Calculate properties using MACE
        self.mace_calc.calculate(tmp_atoms, properties, system_changes)

        # Restore the original atomic numbers and types
        tmp_atoms.set_atomic_numbers(original_atom_numbers)
        tmp_atoms.arrays['atom_types'] = self.atom_types

        # Copy results from MACE calculator
        self.results = self.mace_calc.results.copy() 