import os
import sys
import numpy as np
import pandas as pd
from pymatgen.io.vasp import Poscar
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from scipy.spatial import Voronoi
from data_processing import get_bond_indices, calc_pair_wise_properties
import warnings
warnings.filterwarnings("ignore") # ignore RuntimeWarning (division by zero)


def set_structure_from_poscar_file(POSCAR_path):
    poscar = Poscar.from_file(POSCAR_path, check_for_POTCAR=False,
                            read_velocities=False)
    structure = poscar.structure
    return structure


# Compute adjacency matrix A.
def get_adj_matrix(structure, tol=0, cutoff=10):
    n_atoms = len(structure)
    adj_matrix_no_weight = np.zeros((n_atoms, n_atoms))
    adj_matrix_multi_edge = np.zeros((n_atoms, n_atoms))
    adj_matrix_sol_angle = np.zeros((n_atoms, n_atoms))

    # compute adjacency matrix
    voro = VoronoiNN(tol=tol, cutoff=cutoff, allow_pathological=True,
                      extra_nn_info=False, compute_adj_neighbors=False)
    all_nn_info = voro.get_all_nn_info(structure)
    # we force a graph not to have self-loops
    for center_index in range(n_atoms):
        for nn_index in range(len(all_nn_info[center_index])):
            nn_site_index = all_nn_info[center_index][nn_index]['site_index']
            if nn_site_index > center_index:
                # weight_type: None
                adj_matrix_no_weight[center_index, nn_site_index] = 1
                # weight_type: 'multi_edge'
                adj_matrix_multi_edge[center_index, nn_site_index] += 1
                # weight_type: 'sol_angle':
                sol_angle = all_nn_info[center_index][nn_index]['weight']
                adj_matrix_sol_angle[center_index, nn_site_index] += sol_angle
            else:
                adj_matrix_no_weight[center_index, nn_site_index] = \
                        adj_matrix_no_weight[nn_site_index, center_index]
                adj_matrix_multi_edge[center_index, nn_site_index] = \
                        adj_matrix_multi_edge[nn_site_index, center_index]
                adj_matrix_sol_angle[center_index, nn_site_index] = \
                        adj_matrix_sol_angle[nn_site_index, center_index]

    return adj_matrix_no_weight, adj_matrix_multi_edge, adj_matrix_sol_angle


# calculate Laplacian matrix from adjacency matrix
def get_laplacian(adj_matrix, structure, potential=False):
    # compute degree matrix
    deg_matrix = np.diag(np.sum(adj_matrix, axis=1))
    # standard Laplacian
    laplacian = deg_matrix - adj_matrix
    # generalized laplacian (with potential)
    if potential == True:
        distance_matrix = structure.distance_matrix
        reciprocal_distance_matrix = distance_matrix ** -1
        reciprocal_distance_matrix[reciprocal_distance_matrix == np.inf] = 0
        potential_matrix = np.diag(np.sum(reciprocal_distance_matrix, axis=1))
        laplacian = laplacian - potential_matrix

    return laplacian


def compute_descriptors_from_laplacian(laplacian):
    # compute eigenvalues and eigenvectors
    w, v = np.linalg.eigh(laplacian)
    # It is sure that the nontrivial smallest eigenvalue index is in the second
    # since all graphs are connected graph.
    nontrivial_smallest_eigenvalue_index = np.argsort(w)[1]
    alg_connectivity = w[nontrivial_smallest_eigenvalue_index]

    return alg_connectivity, np.mean(w), np.std(w)


# compute Laplacian-based descriptors
def calc_laplacian_descriptors(atomic_df, POSCAR_path, tol=0, cutoff=10):
    structure = set_structure_from_poscar_file(POSCAR_path)
    primitive_structure = structure.get_primitive_structure()
    species = primitive_structure.species

    adj_matrix_no_weight, adj_matrix_multi_edge, adj_matrix_sol_angle = \
                            get_adj_matrix(primitive_structure, tol, cutoff)

    bond_indices = get_bond_indices(adj_matrix_no_weight)
    diff_mat = calc_pair_wise_properties(atomic_df, bond_indices, species, 'diff')

    adj_matrix_atomic_weighted = diff_mat * adj_matrix_no_weight
    # shape => (n_properties, n_atoms, n_atoms)
    adj_matrix_atomic_multi_edge = diff_mat * adj_matrix_multi_edge
    # shape => (n_properties, n_atoms, n_atoms)

    adj_matrix_list = [
            adj_matrix_no_weight, adj_matrix_multi_edge, adj_matrix_sol_angle]
    all_adj_matrix_list = adj_matrix_list + \
                        adj_matrix_atomic_weighted.tolist() + \
                        adj_matrix_atomic_multi_edge.tolist()
    laplacian_list = [get_laplacian(np.array(x), structure, potential=True)
                            for x in all_adj_matrix_list]
    descriptors = np.array([compute_descriptors_from_laplacian(laplacian)
                            for laplacian in laplacian_list])
    # Even if a graph is weighted, the Fiedler vector is the same with
    # one without weight. Therefore we calculate the std of the Fiedler vector
    # only once.
    w, v = np.linalg.eigh(laplacian_list[0])
    nontrivial_smallest_eigenvalue_index = np.argsort(w)[1]
    std_fiedler_vector = np.std(v[:, nontrivial_smallest_eigenvalue_index])
    descriptors = np.append(descriptors, std_fiedler_vector)

    return descriptors
