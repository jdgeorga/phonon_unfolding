import numpy as np
from sklearn.cluster import DBSCAN
from ase.atoms import Atoms

def proj_phonons(phonon, phonon_eigvecs):
    unitcell = phonon.unitcell.cell
    atom_pos = phonon.primitive.positions
    atom_type = phonon.primitive.numbers
    atom_masses = phonon.primitive.masses

    unit_atoms = len(atom_pos)
    atom_list = np.array(np.arange(unit_atoms), dtype=int)
    atom_xyz_long = atom_list.repeat(3) + np.tile(np.arange(3), unit_atoms) * 6
    atom_symbol_plot = np.where(np.logical_or(atom_type == 42, atom_type == 74), "s", "o")
    layer_index = np.where(np.logical_and(atom_pos[:, 2] > atom_pos[:, 2].mean(), atom_pos[:, 2] < unitcell[2][2] / 2), 1, 0)

    vec_all = np.abs(np.array(phonon_eigvecs).reshape(-1, unit_atoms * 3, unit_atoms * 3))

    layer_character = np.zeros_like(vec_all[:, :, 0])
    plane_character = np.zeros_like(vec_all[:, :, 0])
    band_character = np.zeros_like(vec_all[:, :, 0])

    proj_layer = np.zeros((len(vec_all), len(atom_pos) * 3, 2))
    proj_plane = np.zeros((len(vec_all), len(atom_pos) * 3, 2))

    print("Mo:0, W:1")
    print("In:0, Out:1")
    for i in np.arange(len(atom_pos) * 3):
        for q in np.arange(len(vec_all)):
            m1 = vec_all[q, :, i]
            bottom_mask = np.zeros(len(atom_pos) * 3)
            bottom_mask[:int(len(atom_pos) * 3 / 2)] = 1.

            top_mask = np.ones(len(atom_pos) * 3)
            top_mask[:int(len(atom_pos) * 3 / 2)] = 0.

            proj_layer[q, i, 0] = np.abs(m1) ** 2 @ bottom_mask
            proj_layer[q, i, 1] = np.abs(m1) ** 2 @ top_mask

            in_mask = np.ones((len(atom_pos), 3))
            in_mask[:, 2] = 0.
            in_mask = in_mask.reshape(-1)

            out_mask = np.zeros((len(atom_pos), 3))
            out_mask[:, 2] = 1.
            out_mask = out_mask.reshape(-1)

            proj_plane[q, i, 0] = np.abs(m1) ** 2 @ in_mask
            proj_plane[q, i, 1] = np.abs(m1) ** 2 @ out_mask

    return proj_layer, proj_plane

def add_allegro_number_array(ase_atom: Atoms, eps: float = .75, min_samples: int = 1) -> np.ndarray:
    """
    Assigns a number from 0 to 5 to each atom depending on its layer and position using DBSCAN clustering algorithm.

    :param ase_atom: ASE Atoms object with atom positions.
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: A numpy array with assigned cluster numbers for each atom.
    """
    z_coords = ase_atom.positions[:, 2].reshape(-1, 1)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(z_coords)
    labels = db.labels_

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]

    mean_positions = [z_coords[labels == label].mean() for label in unique_labels]
    sorted_indices = np.argsort(mean_positions)

    label_mapping = np.array([1, 2, 0, 4, 5, 3])

    allegro_number_array = np.array([sorted_indices[label_mapping][label] if label != -1 else -1 for label in labels])

    return allegro_number_array

def proj_phonons_nlayer(atoms, phonon, phonon_eigvecs, atom_layers):
        """
        Project phonon eigenvectors onto different layers.

        :param phonon: Phonopy object with the phonon band structure.
        :param phonon_eigvecs: Phonon eigenvectors.
        :param atom_layers: Array indicating the layer assignment of each atom.
        :return: Projections onto layers and planes.
        """
        atom_layers = np.array(atom_layers)
        num_layers = len(np.unique(atom_layers))
        phonon_eigvecs = phonon.get_band_structure_dict()['eigenvectors'][0]
        unitcell = phonon.unitcell.cell
        atom_pos = phonon.primitive.positions
        atom_type = phonon.primitive.numbers
        atom_masses = phonon.primitive.masses

        unit_atoms = len(atom_pos)
        layer_index = atom_layers[atoms.arrays['atom_types']]
        vec_all = np.abs(np.array(phonon_eigvecs).reshape(-1, unit_atoms * 3, unit_atoms * 3))

        proj_layer = np.zeros((len(vec_all), len(atom_pos) * 3, 3))
        proj_plane = np.zeros((len(vec_all), len(atom_pos) * 3, 3))

        color_index = list(range(len(np.unique(layer_index))))

        for i in np.arange(len(atom_pos) * 3):
            for q in np.arange(len(vec_all)):
                m1 = vec_all[q, :, i]

                for l in np.arange(len(np.unique(layer_index))):                
                    layer_mask = np.zeros((len(atom_pos) * 3))
                    layer_mask[layer_index.repeat(3) == l] = 1.

                    proj_layer[q, i, color_index[l]] = np.abs(m1) ** 2 @ layer_mask

                    in_mask = np.ones((len(atom_pos), 3))
                    in_mask[:, 2] = 0.
                    in_mask = in_mask.reshape(-1)

                    out_mask = np.zeros((len(atom_pos), 3))
                    out_mask[:, 2] = 1.
                    out_mask = out_mask.reshape(-1)

                    proj_plane[q, i, 0] = np.abs(m1) ** 2 @ in_mask
                    proj_plane[q, i, 1] = np.abs(m1) ** 2 @ out_mask

        return proj_layer, proj_plane

def phonopy_atoms_to_ase(atoms_phonopy):
    """Convert PhonopyAtoms to ASE Atoms."""
    from ase.atoms import Atoms

    ase_atoms = Atoms(
        cell=atoms_phonopy.cell,
        scaled_positions=atoms_phonopy.scaled_positions,
        numbers=atoms_phonopy.numbers,
        pbc=True,
    )
    return ase_atoms

def gen_ase(cell, scaled_positions, numbers):
    """Convert cell, scaled_positions, and numbers to ASE Atoms."""
    from ase.atoms import Atoms

    ase_atoms = Atoms(
        cell=cell,
        scaled_positions=scaled_positions,
        numbers=numbers,
        pbc=True,
    )
    return ase_atoms

def ase_to_phonopy_atoms(ase_atoms):
    """Convert ASE Atoms to PhonopyAtoms."""
    from phonopy.structure.atoms import PhonopyAtoms

    phonopy_atoms = PhonopyAtoms(
        symbols=ase_atoms.get_chemical_symbols(),
        scaled_positions=ase_atoms.get_scaled_positions(),
        cell=ase_atoms.cell,
    )
    return phonopy_atoms
