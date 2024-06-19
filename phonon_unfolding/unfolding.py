import numpy as np
from scipy.spatial import cKDTree
from phonon_unfolding.kpoints import generate_mBZ_multiples, build_primitive_bz_grid

def get_Q_G_b_list_and_cart_Q_list(path_qpoints, primitive_reciprocal_vectors, supercell_reciprocal_vectors, limit=3):
    """
    Generate Q_G_b_list and cart_Q_list using KDTree.

    :param path_qpoints: List of interpolated q-points segments.
    :param primitive_reciprocal_vectors: 3x3 array-like of primitive reciprocal lattice vectors.
    :param supercell_reciprocal_vectors: 3x3 array-like of supercell reciprocal lattice vectors.
    :param limit: Integer defining the range of multiples to generate.
    :return: Tuple containing Q_G_b_list, cart_Q_list, and Q_points.
    """
    mBZ_multiples = generate_mBZ_multiples(supercell_reciprocal_vectors, limit)
    tree = cKDTree(mBZ_multiples)

    Q_G_b_list = []
    cart_Q_list = []
    inv_supercell_reciprocal = np.linalg.inv(supercell_reciprocal_vectors)
    
    for segment in path_qpoints:
        Q_G_b_segment = []
        cart_Q_segment = []
        
        for q in segment:
            # Convert q to Cartesian coordinates
            q_cartesian = np.dot(q, primitive_reciprocal_vectors)
            
            # Find the nearest multiple of the mBZ
            distance, index = tree.query(q_cartesian)
            G_b_cartesian = tree.data[index]
            
            # Calculate Q
            Q_cartesian = q_cartesian - G_b_cartesian
            Q = np.dot(Q_cartesian, inv_supercell_reciprocal)
            
            # Convert G_b_cartesian back to crystal coordinates
            G_b = np.dot(G_b_cartesian, inv_supercell_reciprocal)

            cart_Q_segment.append((q_cartesian, Q_cartesian, G_b_cartesian))
            Q_G_b_segment.append((q, Q, G_b))
        
        Q_G_b_list.append(Q_G_b_segment)
        cart_Q_list.append(cart_Q_segment)
    
    Q_points = [[Q for _, Q, _ in segment] for segment in Q_G_b_list]
    
    return Q_G_b_list, cart_Q_list, Q_points

def compute_coefficients(frequencies, eigenvectors, cart_Q_list, primitive_reciprocal_vectors, phonon, grid_size=8):
    """
    Compute the coefficients for the unfolded band structure.

    :param frequencies: Array of phonon frequencies.
    :param eigenvectors: Array of phonon eigenvectors.
    :param cart_Q_list: List of Cartesian Q points.
    :param primitive_reciprocal_vectors: 3x3 array-like of primitive reciprocal lattice vectors.
    :param phonon: Phonopy object.
    :param grid_size: Integer defining the grid size in the primitive BZ.
    :return: List of coefficients.
    """
    num_segments = frequencies.shape[0]
    num_bands = frequencies.shape[2]

    g_j = build_primitive_bz_grid(primitive_reciprocal_vectors, grid_size)
    R_I = phonon.primitive.positions

    coefficients = []
    for segment_idx, segment in enumerate(cart_Q_list):
        coefficients_segment = []
        for q_idx, (q, Q, G_b) in enumerate(segment):
            eigenvectors_Q = eigenvectors[segment_idx][q_idx]
            eigenvectors_Q = eigenvectors_Q.reshape(len(phonon.primitive.positions), 3, len(phonon.primitive.positions) * 3)
            N = len(R_I)
            B_js = compute_B_js(Q, G_b, eigenvectors_Q, R_I, g_j, N)
            c_Q_G_b_n = compute_c_Q_G_b_n(B_js)
            coefficients_segment.append(c_Q_G_b_n / np.shape(g_j)[0])
        coefficients.append(np.array(coefficients_segment))

    return coefficients

def compute_qs_cart(cart_Q_list):
    """
    Compute the Cartesian q-points for the unfolded band structure.

    :param cart_Q_list: List of Cartesian Q points.
    :return: List of Cartesian q-points.
    """
    qs_cart = []
    for segment_idx, segment in enumerate(cart_Q_list):
        qs_segment = []
        for q_idx, (q, Q, G_b) in enumerate(segment):
            qs_segment.append(q)
        qs_cart.append(qs_segment)

    return qs_cart

def compute_B_js(Q, G_b, eigenvectors_Q, R_I, g_j, N):
    """
    Compute the B_js for unfolding.

    :param Q: Q point.
    :param G_b: G_b vector.
    :param eigenvectors_Q: Array of eigenvectors at Q.
    :param R_I: Array of atomic positions.
    :param g_j: Grid points in the primitive BZ.
    :param N: Number of atoms.
    :return: Array of B_js.
    """
    num_j = len(g_j)
    num_s = eigenvectors_Q.shape[1]
    num_n = eigenvectors_Q.shape[2]

    B_js = np.zeros((num_j, num_s, num_n), dtype=complex)

    phases = np.exp(-1j * np.einsum('ji,li->jl', G_b + g_j, R_I))

    for j, g in enumerate(g_j):
        B_js[j] = np.einsum('i,isn->sn', phases[j], eigenvectors_Q, optimize=True)

    B_js /= np.sqrt(N)

    return B_js

def compute_c_Q_G_b_n(B_js):
    """
    Compute the unfolded coefficients |c_{Q + G_b, n}|^2.

    :param B_js: Array of B_js.
    :return: Array of unfolded coefficients.
    """
    c_Q_G_b_n = np.sum(np.abs(B_js) ** 2, axis=(0, 1))
    return c_Q_G_b_n
