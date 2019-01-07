import os
import sys
import numpy as np
import pandas as pd
from pymatgen.io.vasp import Poscar
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from scipy.spatial import Voronoi
from utils import get_bond_indices, calc_pair_wise_properties
import warnings
warnings.filterwarnings("ignore") # ignore RuntimeWarning (division by zero)


valence_dict = {'Li':1, 'Na':1, 'K':1, 'Rb':1, 'Cs':1,
                'Be':2, 'Mg':2, 'Ca':2, 'Sr':2, 'Ba':2, 'Zn':2, 'Cd':2, 'Hg':2,
                'Al':3, 'Ga':3, 'In':3, 'Sc':3, 'Y':3, 'La':3,
                'F':-1, 'Cl':-1, 'Br':-1, 'I':-1,
                'O':-2, 'S':-2, 'Se':-2, 'Te':-2,
                'N':-3, 'P':-3, 'As':-3, 'Sb':-3}


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
def get_laplacian(adj_matrix, structure, potential=True):
    # compute degree matrix
    deg_matrix = np.diag(np.sum(adj_matrix, axis=1))
    # standard Laplacian
    laplacian = deg_matrix - adj_matrix
    # generalized laplacian (with potential)
    if potential == True:
        species = structure.species
        coulomb_matrix = np.zeros((len(species), len(species)))
        for i in range(len(species)):
            for j in range(len(species)):
                if i < j:
                    coulomb_matrix[i, j] = valence_dict[str(species[i])] * \
                                            valence_dict[str(species[j])]
                else:
                    coulomb_matrix[i, j] = coulomb_matrix[j, i]
        distance_matrix = structure.distance_matrix
        reciprocal_distance_matrix = distance_matrix ** -1
        reciprocal_distance_matrix[reciprocal_distance_matrix == np.inf] = 0
        coulomb_matrix = coulomb_matrix * reciprocal_distance_matrix
        potential_matrix = np.diag(np.sum(coulomb_matrix, axis=1))
        # potential_matrix = np.diag(np.sum(reciprocal_distance_matrix, axis=1))
        laplacian = laplacian + potential_matrix

    return laplacian


def compute_descriptors_from_laplacian(laplacian):
    # compute eigenvalues and eigenvectors
    w, v = np.linalg.eigh(laplacian)
    # It is sure that the nontrivial smallest eigenvalue index is in the second
    # since all graphs are connected graph.
    # nontrivial_smallest_eigenvalue_index = np.argsort(w)[1]
    # alg_connectivity = w[nontrivial_smallest_eigenvalue_index]

    return np.mean(w), np.std(w)


# compute Laplacian-based descriptors
def calc_laplacian_descriptors(atomic_df, POSCAR_path, adj_matrix_list=None,
                                tol=0, cutoff=10, potential=True):
    structure = set_structure_from_poscar_file(POSCAR_path)
    primitive_structure = structure.get_primitive_structure()
    species = primitive_structure.species

    if adj_matrix_list == None:
        adj_matrix_no_weight, adj_matrix_multi_edge, adj_matrix_sol_angle = \
                            get_adj_matrix(primitive_structure, tol, cutoff)
    else:
        adj_matrix_no_weight, adj_matrix_multi_edge, adj_matrix_sol_angle = \
                    adj_matrix_list[0], adj_matrix_list[1], adj_matrix_list[2]

    bond_indices = get_bond_indices(adj_matrix_no_weight)
    diff_mat = calc_pair_wise_properties(atomic_df, bond_indices, species, 'diff')

    adj_matrix_atomic_weighted = diff_mat * adj_matrix_no_weight
    # shape => (n_properties, n_atoms, n_atoms)
    adj_matrix_atomic_multi_edge = diff_mat * adj_matrix_multi_edge
    # shape => (n_properties, n_atoms, n_atoms)
    adj_matrix_atomic_sol_angle = diff_mat * adj_matrix_sol_angle

    adj_matrix_list = [
            adj_matrix_no_weight, adj_matrix_multi_edge, adj_matrix_sol_angle]
    all_adj_matrix_list = adj_matrix_list + \
                        adj_matrix_atomic_weighted.tolist() + \
                        adj_matrix_atomic_multi_edge.tolist() + \
                        adj_matrix_atomic_sol_angle.tolist()
    laplacian_list = [get_laplacian(np.array(x),
                        primitive_structure, potential=potential)
                            for x in all_adj_matrix_list]
    descriptors = np.array([compute_descriptors_from_laplacian(laplacian)
                            for laplacian in laplacian_list])
    # Even if a graph is weighted, the Fiedler vector is the same with
    # one without weight. Therefore we calculate the std of the Fiedler vector
    # only once.
    # w, v = np.linalg.eigh(laplacian_list[0])
    # nontrivial_smallest_eigenvalue_index = np.argsort(w)[0]
    # mean_fiedler_vector = np.mean(v[:, nontrivial_smallest_eigenvalue_index])
    # std_fiedler_vector = np.std(v[:, nontrivial_smallest_eigenvalue_index])
    # nontrivial_smallest_eigenvalue_index2 = np.argsort(w)[1]
    # mean_fiedler_vector2 = np.mean(v[:, nontrivial_smallest_eigenvalue_index2])
    # std_fiedler_vector2 = np.std(v[:, nontrivial_smallest_eigenvalue_index2])
    # descriptors = np.append(descriptors, mean_fiedler_vector)
    # descriptors = np.append(descriptors, std_fiedler_vector)
    # descriptors = np.append(descriptors, mean_fiedler_vector2)
    # descriptors = np.append(descriptors, std_fiedler_vector2)

    return descriptors.flatten()
