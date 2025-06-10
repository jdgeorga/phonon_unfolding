import numpy as np

def reciprocal_lattice_vectors(lattice_vectors):
    """
    Calculate reciprocal lattice vectors from given lattice vectors.

    :param lattice_vectors: 3x3 array-like containing the lattice vectors.
    :return: 3x3 numpy array containing the reciprocal lattice vectors.
    """
    a1, a2, a3 = lattice_vectors
    volume = np.dot(a1, np.cross(a2, a3))
    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume
    b3 = 2 * np.pi * np.cross(a1, a2) / volume
    return np.array([b1, b2, b3])

def interpolate_qpoints(Qpoints, num_points):
    """
    Interpolate q-points along a high-symmetry path.

    :param Qpoints: List or array-like of high-symmetry q-points.
    :param num_points: Number of points to interpolate between each pair of q-points.
    :return: List of interpolated q-points segments.
    """
    path_qpoints = []
    for i in range(len(Qpoints) - 1):
        start = Qpoints[i]
        end = Qpoints[i + 1]
        print(i, start, end)
        segment = np.linspace(start, end, num_points, endpoint=True)  # Include endpoint
        path_qpoints.append(segment.tolist())
    return path_qpoints

def generate_mBZ_multiples(supercell_reciprocal_vectors, limit):
    """
    Generate multiples of the mini Brillouin zone (mBZ).

    :param supercell_reciprocal_vectors: 3x3 array-like of supercell reciprocal lattice vectors.
    :param limit: Integer defining the range of multiples to generate.
    :return: Numpy array of mBZ multiples.
    """
    multiples = []
    for i in range(-limit*7, 7*limit+1):
        for j in range(-limit, limit+1):
            G_b = i * supercell_reciprocal_vectors[0] + j * supercell_reciprocal_vectors[1]
            multiples.append(G_b)
    return np.array(multiples)

def get_high_symmetry_qpoints(atoms):
    """
    Get high-symmetry q-points and their labels for a given set of atoms.

    :param atoms: ASE Atoms object.
    :return: Tuple containing high-symmetry q-points and their labels.
    """
    bp = atoms.cell.bandpath(npoints=0, pbc=[1, 1, 0])

    high_symmetry_q_points = [x for x in bp.special_points.values()]
    high_symmetry_q_points.append(high_symmetry_q_points[0])
    high_symmetry_q_points = np.array(high_symmetry_q_points)
    high_symmetry_labels = [x for x in bp.special_points.keys()]
    high_symmetry_labels.append(high_symmetry_labels[0])

    return high_symmetry_q_points, high_symmetry_labels

def build_primitive_bz_grid(primitive_reciprocal_vectors, grid_size):
    """
    Build a grid in the primitive Brillouin zone (BZ).

    :param primitive_reciprocal_vectors: 3x3 array-like of primitive reciprocal lattice vectors.
    :param grid_size: Integer defining the grid size.
    :return: Numpy array of grid points in the primitive BZ.
    """
    grid_points = []
    for i in range(-grid_size, grid_size + 1):
        for j in range(-grid_size, grid_size + 1):
            point = i * primitive_reciprocal_vectors[0] + j * primitive_reciprocal_vectors[1]
            grid_points.append(point)
    return np.array(grid_points)
