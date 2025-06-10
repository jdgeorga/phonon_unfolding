
# Moiré Phonon Analysis

This repository contains three scripts to analyze the phonon band structure of MoS2 1D moiré patterns. The scripts are designed to work with MPI for parallel processing.

## Scripts

1. **1_generate_phonopy_yaml.py**
   - Generates a Phonopy YAML file with displacement information from a relaxed atomic structure.

2. **2_generate_forces.py**
   - Uses MPI to generate forces for each displacement structure using Allegro or LAMMPS calculators.

3. **3_analyze_structure.py**
   - Analyzes the structure to produce and plot the unfolded phonon band structure based on the forces and the Phonopy YAML file.

## Setup

### Prerequisites

- Python 3.x
- ASE
- Phonopy
- mpi4py
- numpy
- matplotlib

### Installation

1. Clone the repository:
    \`\`\`bash
    git clone https://github.com/your-repository/MoS2-1D-Phonon-Analysis.git
    cd MoS2-1D-Phonon-Analysis
    \`\`\`

2. Install the required Python packages:
    \`\`\`bash
    pip install numpy ase phonopy mpi4py matplotlib
    \`\`\`

### Files and Directories

- **relaxed_structure.xyz**: Relaxed supercell structure file.
- **primitive_unitcell.xyz**: Primitive unit cell structure file.
- **intralayer_allegro.pth**: Intralayer model file for Allegro calculator.
- **interlayer_allegro.pth**: Interlayer model file for Allegro calculator.

## Usage

### 1. Generate Phonopy YAML File

This script generates a Phonopy YAML file with displacement information.

\`\`\`bash
python 1_generate_phonopy_yaml.py
\`\`\`

**Expected Output:**

- \`phonon_with_displacements.yaml\`: YAML file containing Phonopy displacement information.
- Printed displacements.

### 2. Generate Forces

This script uses MPI to generate forces for each displacement structure. Run it using the \`srun\` command with the desired number of processes.

\`\`\`bash
srun -N 1 -n (num processes) python 2_generate_forces.py
\`\`\`

**Example:**

\`\`\`bash
srun -N 1 -n 4 python 2_generate_forces.py
\`\`\`

**Expected Output:**

- \`disp_forces.npy\`: Numpy file containing displacement forces.

### 3. Analyze Structure

This script analyzes the structure to produce and plot the unfolded phonon band structure.

\`\`\`bash
python 3_analyze_structure.py
\`\`\`

**Expected Output:**
- \`folded_phonon_band_structure.png\`: Plot folded supercell phonon band structure.
- \`unfold_phonon_band_structure.png\`: Plot of the unfolded phonon band structure.
- \`Path_on_Wigner_Seitz_cells.png\`: Plot of the high-symmetry paths.
- Printed steps during the analysis process.

## Description of Each Script

### 1_generate_phonopy_yaml.py

This script generates a Phonopy YAML file with displacement information from a relaxed atomic structure.

**Arguments:**
- \`input_file\`: Path to the input file containing the relaxed atomic structure.
- \`supercell_dim\`: List of three integers representing the supercell dimensions.

### 2_generate_forces.py

This script uses MPI to generate forces for each displacement structure using Allegro or LAMMPS calculators.

**Arguments:**
- \`input_file\`: Path to the input file containing the relaxed atomic structure.
- \`yaml_file\`: Path to the Phonopy YAML file with displacements.
- \`layer_symbols\`: List of layer symbols.
- \`calculator_type\`: Type of calculator to use (allegro or lammps).
- \`out_file\`: Path to the output file to save the forces.
- \`intralayer_model\`: Path to the intralayer model file (only for Allegro).
- \`interlayer_model\`: Path to the interlayer model file (only for Allegro).

### 3_analyze_structure.py

This script analyzes the structure to produce and plot the unfolded phonon band structure based on the forces and the Phonopy YAML file.

**Arguments:**
- \`input_file\`: Path to the input file containing the relaxed atomic structure.
- \`disp_forces_file\`: Path to the input file containing the displacement forces (numpy format).
- \`phonopy_yaml_file\`: Path to the input file containing the Phonopy YAML data.
- \`primitive_file\`: Path to the input file containing the primitive unit cell structure.
- \`atom_layers\`: List of integers indicating the layer assignment of each atom.

---