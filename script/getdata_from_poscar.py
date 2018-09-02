import os
import sys
import numpy as np
import pandas as pd
from pymatgen.io.vasp import Poscar
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from scipy.spatial import Voronoi
from itertools import chain
from numpy import inf
import warnings

dir_path = os.getcwd()
atomic_df = pd.read_csv(dir_path + "/data/atomic_data_reduced.csv", index_col=0)
Cordero_cov_radii = atomic_df['Rcov']


class DataConstruction:

    def __init__(self, tol=1e-4, cutoff=10.0, cov_radii_tol=0.65,
                multiplication_type="element_wise", connectivity="distance"):
        '''
        tol (float):
                tolerance parameter for bond determination. (default: 1e-4)

        cutoff (float):
                cutoff radius in Angstrom to look for near-neighbour atoms.
                (default: 10.0)

        cov_radii_tol (float):
                The interatomic distance must be shorter than the sum of the
                Cordero covalent radii to within this tolerance. The unit is
                angstrom. This param is ignored when param connectivity is set
                to "weight".

        multiplication_type (str):
                how to multiply matrices. element-wise or matrix multiplication.
                (default: "element_wise")

        connectivity (str):
                how to define the connectivity. If set to "distance", for all
                bonds, the interatomic distance must be shorter than the sum of
                the Cordero covalent radii to within cov_radii_tol. If set to
                "weight", elements of the adjacency matrix will be the sum of
                connectivity weight, which is the normalized solid angle
                (= solid andgle / maximal solid angle).
        '''
        self.tol= tol
        self.cutoff = cutoff
        self.cov_radii_tol = cov_radii_tol
        self.multiplication_type = multiplication_type
        self.connectivity = connectivity


    def get_info_from_poscar(self, f):
        lines = f.readlines()
        # lattice vectors
        a = float(lines[1]) # a: scale
        v1 = a * np.array(list(map(float, lines[2].strip().split())))
        v2 = a * np.array(list(map(float, lines[3].strip().split())))
        v3 = a * np.array(list(map(float, lines[4].strip().split())))
        lattice = np.array([v1, v2, v3])

        # list of elements
        elements = list(map(str, lines[5].strip().split()))
        # the number of atoms in unit cell per element
        n_atom_each_ele = list(map(int, lines[6].strip().split()))
        n_atoms = sum(n_atom_each_ele)
        species_list = [[elements[i]]*n_atom_each_ele[i] \
                            for i in range(len(elements))]
        species = list(chain.from_iterable(species_list))

        # coordinates of each point
        points = []
        for i in range(8, 8+n_atoms):
            points.append(list(map(float, lines[i].strip().split())))
        coords = np.array(points)

        return (lattice, species, coords)


    def set_structure_from_poscar_info(self, POSCAR_path):
        with open(POSCAR_path) as f:
            lattice, species, coords = self.get_info_from_poscar(f)
            structure = Structure(lattice, species, coords)
            n_atoms = len(structure)

            return (structure, n_atoms, species)


    def set_structure_from_poscar_file(self, POSCAR_path):
        poscar = Poscar.from_file(POSCAR_path, check_for_POTCAR=False,
                                read_velocities=False)
        structure = poscar.structure

        return structure


    # Compute adjacency matrix A and reciprocal squared distance matrix D.
    # Periodic boundary condition is considered.
    def voronoi_periodic(self, structure):
        n_atoms = len(structure)
        voro = VoronoiNN(tol=self.tol, cutoff=self.cutoff, allow_pathological=True)
        all_nn_info = [voro.get_nn_info(structure, i) for i in range(n_atoms)]

        # The following setting is for avoiding the unavailability of
        # adjacency matrix that occurs when an input is a simple substance.
        if len(structure) == 1:
            adjacency_matrix = np.zeros((2, 2))
        else:
            adjacency_matrix = np.zeros((n_atoms, n_atoms))

        # compute adjacency matrix
        for center_index in range(n_atoms):
            center_site = structure[center_index]
            center_element = center_site.species_string

            for nn_index in range(len(all_nn_info[center_index])):
                nn_site_index = all_nn_info[center_index][nn_index]["site_index"]

                if nn_site_index >= center_index:
                    if self.connectivity == "distance":
                        nn_site = all_nn_info[center_index][nn_index]["site"]
                        nn_element = nn_site.species_string
                        distance = np.linalg.norm(center_site.coords - nn_site.coords)

                        if (distance <= Cordero_cov_radii[center_element] +
                                        Cordero_cov_radii[nn_element] +
                                        float(self.cov_radii_tol)):
                            if len(structure) == 1:
                                adjacency_matrix[0,1] += 1
                            else:
                                adjacency_matrix[center_index, nn_site_index] += 1

                    elif self.connectivity == "weight":
                        weight = all_nn_info[center_index][nn_index]["weight"]
                        if len(structure) == 1:
                            adjacency_matrix[0,1] += weight
                        else:
                            adjacency_matrix[center_index, nn_site_index] += weight

                else:
                    adjacency_matrix[center_index, nn_site_index] = \
                            adjacency_matrix[nn_site_index, center_index]

        if len(structure) == 1:
            adjacency_matrix[1,0] = adjacency_matrix[0,1]
            squared_distance_matrix = np.zeros((2,2))
            squared_distance_matrix[0,1] = distance ** 2
            squared_distance_matrix[1,0] = distance ** 2
        else:
            squared_distance_matrix = structure.distance_matrix ** 2

        warnings.filterwarnings("ignore") # ignore RuntimeWarning (division by zero)
        reciprocal_squared_distance_matrix = squared_distance_matrix ** -1
        reciprocal_squared_distance_matrix[reciprocal_squared_distance_matrix == inf]=0

        return (adjacency_matrix, reciprocal_squared_distance_matrix)


    # Compute adjacency matrix A and reciprocal squared distance matrix D.
    # Periodic boundary condition is NOT considered.
    def voronoi_Nonperiodic(self, structure):
        """
        VoronoiConnectivity().get_connections()

        input:
        structure - (Structure)
        cutoff - (float)

        return:
        A list of site pairs that are Voronoi Neighbours,
        along with their real-space distances.

        The shape of a list is N×3, where N is the number of pairs.
        e.g. [[0, 1, 3.9710929317605546], [0, 2, 3.971092931760555],
              [0, 3, 3.971092931760555], ...]
        """
        n_atoms = len(structure)
        connections = VoronoiConnectivity(structure, cutoff=self.cutoff).get_connections()
        adjacency_matrix = np.zeros((n_atoms, n_atoms))

        # compute adjacency matrix
        for n in range(len(connections)):
            site_i = connections[n][0]
            site_j = connections[n][1]
            adjacency_matrix[site_i, site_j] += 1

        # compute squared distance matrix
        squared_distance_matrix = structure.distance_matrix ** 2

        warnings.filterwarnings("ignore") # ignore RuntimeWarning (division by zero)
        reciprocal_squared_distance_matrix = squared_distance_matrix ** -1
        reciprocal_squared_distance_matrix[reciprocal_squared_distance_matrix == inf] = 0

        return (adjacency_matrix, reciprocal_squared_distance_matrix)


    # Multiply adjacency matrix and reciprocal squared distance matrix
    # with the chosen multiplication type.
    def multiply_adjacency_and_distance_matrix(self, adjacency_matrix,
                                          reciprocal_squared_distance_matrix):
        if self.multiplication_type == "element_wise":
            multiplied_matrix = adjacency_matrix * reciprocal_squared_distance_matrix
        else: # multiplication_type == "matmul"
            multiplied_matrix = \
                        np.matmul(adjacency_matrix, reciprocal_squared_distance_matrix)

        return multiplied_matrix
