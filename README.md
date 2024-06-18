# Phonon Unfolding

This package provides tools for unfolding phonon band structures and parallel force calculations.

## Installation

To install the package, run:

## Installation

`pip install -e .`


```
from phonon_unfolding.main import main
from ase.io import read
import numpy as np

# Load your structure and displacement forces
atoms = read("path/to/your/structure.xyz")
disp_forces = np.load("path/to/your/disp_forces.npy")

# Call the main function
main(atoms, disp_forces)

```