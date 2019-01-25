import os
import sys
import numpy as np
import pandas as pd
from pymatgen.io.vasp import Poscar
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from scipy.spatial import Voronoi
from laplacian import get_adj_matrix
from utils import get_bond_indices, calc_pair_wise_properties
import warnings
warnings.filterwarnings("ignore") # ignore RuntimeWarning (division by zero)


def set_structure_from_poscar_file(POSCAR_path):
    poscar = Poscar.from_file(POSCAR_path, check_for_POTCAR=False,
                            read_velocities=False)
    structure = poscar.structure
    return structure


# calculate graph descriptors
def calc_graph_descriptors(atomic_df, POSCAR_path, adj_matrix_list=None,
                            tol=0, cutoff=10):
    structure = set_structure_from_poscar_file(POSCAR_path)
    primitive_structure = structure.get_primitive_structure()
    species = primitive_structure.species
    squared_distance_matrix = primitive_structure.distance_matrix ** 2
    reciprocal_squared_distance_matrix = squared_distance_matrix ** -1
    reciprocal_squared_distance_matrix[reciprocal_squared_distance_matrix == np.inf]=0

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

    # no_weight
    ele_wise_mat = adj_matrix_atomic_weighted * reciprocal_squared_distance_matrix
    ele_wise_sum = np.sum(ele_wise_mat, axis=(1,2))
    ele_wise_mean = np.mean(ele_wise_mat, axis=(1,2))
    ele_wise_std = np.std(ele_wise_mat, axis=(1,2))
    matmul_mat = np.matmul(adj_matrix_atomic_weighted,
                            reciprocal_squared_distance_matrix)
    matmul_sum = np.sum(matmul_mat, axis=(1,2))
    matmul_mean = np.mean(matmul_mat, axis=(1,2))
    matmul_std = np.std(matmul_mat, axis=(1,2))

    simple_concat_sum = np.concatenate([ele_wise_sum, matmul_sum])
    simple_concat_mean = np.concatenate([ele_wise_mean, matmul_mean])
    simple_concat_std = np.concatenate([ele_wise_std, matmul_std])

    # weighted by coordination number
    multi_edge_ele_wise_mat = adj_matrix_atomic_multi_edge * reciprocal_squared_distance_matrix
    multi_edge_ele_wise_sum = np.sum(multi_edge_ele_wise_mat, axis=(1,2))
    multi_edge_ele_wise_mean = np.mean(multi_edge_ele_wise_mat, axis=(1,2))
    multi_edge_ele_wise_std = np.std(multi_edge_ele_wise_mat, axis=(1,2))
    multi_edge_matmul_mat = np.matmul(adj_matrix_atomic_multi_edge,
                                    reciprocal_squared_distance_matrix)
    multi_edge_matmul_sum = np.sum(multi_edge_matmul_mat, axis=(1,2))
    multi_edge_matmul_mean = np.mean(multi_edge_matmul_mat, axis=(1,2))
    multi_edge_matmul_std = np.std(multi_edge_matmul_mat, axis=(1,2))

    multi_edge_concat_sum = np.concatenate([multi_edge_ele_wise_sum, multi_edge_matmul_sum])
    multi_edge_concat_mean = np.concatenate([multi_edge_ele_wise_mean, multi_edge_matmul_mean])
    multi_edge_concat_std = np.concatenate([multi_edge_ele_wise_std, multi_edge_matmul_std])

    # weighted by solid angle
    sol_angle_ele_wise_mat = adj_matrix_atomic_sol_angle * reciprocal_squared_distance_matrix
    sol_angle_ele_wise_sum = np.sum(sol_angle_ele_wise_mat, axis=(1,2))
    sol_angle_ele_wise_mean = np.mean(sol_angle_ele_wise_mat, axis=(1,2))
    sol_angle_ele_wise_std = np.std(sol_angle_ele_wise_mat, axis=(1,2))
    sol_angle_matmul_mat = np.matmul(adj_matrix_atomic_sol_angle,
                                    reciprocal_squared_distance_matrix)
    sol_angle_matmul_sum = np.sum(sol_angle_matmul_mat, axis=(1,2))
    sol_angle_matmul_mean = np.mean(sol_angle_matmul_mat, axis=(1,2))
    sol_angle_matmul_std = np.std(sol_angle_matmul_mat, axis=(1,2))

    sol_angle_concat_sum = np.concatenate([sol_angle_ele_wise_sum, sol_angle_matmul_sum])
    sol_angle_concat_mean = np.concatenate([sol_angle_ele_wise_mean, sol_angle_matmul_mean])
    sol_angle_concat_std = np.concatenate([sol_angle_ele_wise_std, sol_angle_matmul_std])

    return simple_concat_sum, \
           simple_concat_mean, \
           simple_concat_std, \
           multi_edge_concat_sum, \
           multi_edge_concat_mean, \
           multi_edge_concat_std, \
           sol_angle_concat_sum, \
           sol_angle_concat_mean, \
           sol_angle_concat_std
