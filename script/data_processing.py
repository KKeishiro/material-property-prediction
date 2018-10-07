import numpy as np
import pandas as pd
from getdata_from_poscar import DataConstruction
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# Get indices from adjacency matrix where a value is nonzero.
def get_bond_indices(adjacency_matrix):
    bond_indices = np.nonzero(adjacency_matrix)
    return bond_indices


# Calculate the pair-wise property values between bond pairs.
def calc_pair_wise_properties(atomic_df, bond_indices, species, calc_type):
    # the shape of pair_mat is n_properties * n_atoms * n_atoms
    assert (calc_type == 'diff' or 'mean'), 'please choose valid calc_type'
    n_atoms = len(species)
    if n_atoms == 1:
        pair_mat = np.zeros((len(atomic_df.columns), 2, 2))
    else:
        pair_mat = np.zeros((len(atomic_df.columns), n_atoms, n_atoms))

    atomic_properties = {}
    species_set = set(species)
    for element in species_set:
        element = element.symbol
        atomic_properties[element] = np.array(atomic_df.loc[element])

    for i in range(len(bond_indices[0])):
        index_i = bond_indices[0][i]
        index_j = bond_indices[1][i]
        if n_atoms == 1:
            atom_i = atomic_properties[species[0].symbol]
            atom_j = atom_i
        else:
            atom_i = atomic_properties[species[index_i].symbol]
            atom_j = atomic_properties[species[index_j].symbol]

        if index_i <= index_j:
            if calc_type == 'diff':
                pair = abs(atom_i - atom_j)
            elif calc_type == 'mean':
                pair = np.mean(np.concatenate([[atom_i], [atom_j]]), axis=0)

            for column_index in range(len(atomic_df.columns)):
                pair_mat[column_index, index_i, index_j] = pair[column_index]
        else:
            for column_index in range(len(atomic_df.columns)):
                pair_mat[column_index, index_i, index_j] = \
                                    pair_mat[column_index, index_j, index_i]

    return pair_mat


# Create descriptors.
def calculate_descriptors(atomic_df, POSCAR_path, tol=1e-4, cutoff=10,
                        cov_radii_tol=0.65,
                        connectivity="distance",
                        multiplication_type="element_wise",
                        isPeriodic=False, descriptor_type="mixtured",
                        max_matrix_power=10,
                        quantize_type_for_structure="trace",
                        quantize_type_for_mixtured="sum"):
    '''
    If it is not possible to calculate Voronoi tessellation,
    you have to increase the value of cutoff. Note that this
    will also increase the calculation time. e.g. cutoff=10 -> cutoff=20

    atomic_df (DataFrame):
            n*d DataFrame, where n is the number of elements and
            d is the number of properties.

    POSCAR_path (str):
            path to POSCAR file.

    tol (float):
            tolerance parameter for bond determination. (default: 0.25)

    cutoff (float):
            cutoff radius in Angstrom to look for near-neighbour atoms.
            When isPeriodic is set True for cohesive energy data, 20
            is used in this work. Otherwise, 10 is used.
            (default: 10.0)

    cov_radii_tol (float):
            The interatomic distance tolerance. This is ignored when
            connectivity is set to "weight". (default: 0.65)

    connnectivity (str):
            The definition of connectivity. You can choose "distance" or
            "weight". (default: "distance")

    multiplication_type (str):
            This defines how to multiply matrices. You can choose either
            element-wise or matrix multiplication. (default: "element_wise")

    isPeriodic (boolean):
            If True, periodic boundary condition is considered.
            Otherwise, it is not considered. (default: False)

    descriptor_type (str):
            This defines what kind of information is contained in descriptors.
            When descriptor_type is "structure", only structure information
            is contained, while when it is "mixtured", you get descriptors
            that contain both atomic information and crystal structure information.
            (default: "mixtured")

    max_matrix_power (int):
            This defines how many descriptors are returned for structure descriptors.
            If descriptor_type is "mixtured", this param is ignored.
            (default: 10)

    quantize_type_for_structure (str):
            This defines how to convert matrix to scaler for structure descriptors.
            You can choose "mean" or "trace" or "std".
            For every case, diagonal elements of matrix are considered.
            If descriptor_type is "mixtured", this param is ignored.
            (default: "trace")

    quantize_type_for_mixtured (str):
            This defines how to convert matrix to scaler for mixtured descriptors.
            You can choose "sum", "mean", "std" and "trace".
            If descriptor_type is "structure", this param is ignored.
            (default: "sum")
    '''
    dc = DataConstruction(tol=tol, cutoff=cutoff, cov_radii_tol=cov_radii_tol,
                        connectivity=connectivity,
                        multiplication_type=multiplication_type)
    structure = dc.set_structure_from_poscar_file(POSCAR_path)
    primitive_structure = structure.get_primitive_structure()
    species = primitive_structure.species

    if isPeriodic == True:
        adjacency_matrix, reciprocal_squared_distance_matrix = \
                                    dc.voronoi_periodic(primitive_structure)
    else:
        adjacency_matrix, reciprocal_squared_distance_matrix = \
                                    dc.voronoi_Nonperiodic(primitive_structure)
    multiplied_matrix = dc.multiply_adjacency_and_distance_matrix(
                        adjacency_matrix, reciprocal_squared_distance_matrix)

    # structure descriptors
    if descriptor_type == "structure":
        descriptors = []
        for i in range(1, max_matrix_power+1):
            if quantize_type_for_structure == "mean":
                descriptor = np.mean(np.linalg.matrix_power(multiplied_matrix, i))
            if quantize_type_for_structure == "trace":
                descriptor = np.trace(np.linalg.matrix_power(multiplied_matrix, i))
            if quantize_type_for_structure == "std":
                descriptor = np.std(np.linalg.matrix_power(multiplied_matrix, i))
            descriptors.append(descriptor)
        # return np.array(descriptors)
        return descriptors

    # mixtured descriptors
    elif descriptor_type == "mixtured":
        bond_indices = get_bond_indices(adjacency_matrix)

        diff_mat = calc_pair_wise_properties(atomic_df, bond_indices, species, 'diff')
        M_diff = diff_mat * multiplied_matrix
        mean_mat = calc_pair_wise_properties(atomic_df, bond_indices, species, 'mean')
        M_mean = mean_mat * multiplied_matrix

        if quantize_type_for_mixtured == "sum":
            diff_descriptor = np.sum(M_diff, axis=(1,2))
            ave_descriptor = np.sum(M_mean, axis=(1,2))
        elif quantize_type_for_mixtured == "mean":
            diff_descriptor = np.mean(M_diff, axis=(1,2))
            ave_descriptor = np.mean(M_mean, axis=(1,2))
        elif quantize_type_for_mixtured == 'std':
            diff_descriptor = np.std(M_diff, axis=(1,2))
            ave_descriptor = np.std(M_mean, axis=(1,2))
        elif quantize_type_for_mixtured == "trace":
            diff_descriptor = np.trace(np.matmul(M_diff, M_diff), axis1=1, axis2=2)
            ave_descriptor = np.trace(np.matmul(M_mean, M_mean), axis1=1, axis2=2)

        a = np.append(diff_descriptor, ave_descriptor)
        # return b
        return a.tolist()


# Normalization by the number of atoms in a unitcell.
def normalization(X_df, encoded_df, save_path):
    '''
    X_df (DataFrame):
            target DataFrame that will be normalized.

    encoded_df (DataFrame):
            corresponding DataFrame to X_df.
            Each raw represents compound. Each column represents element.
            Value in the DataFrame is the number of atoms of corresponding
            element in a unit cell.

    save_path (str):
            path for saving normalized data.
    '''
    for i in range(len(X_df.index)):
        if X_df.index[i] != encoded_df.index[i]:
            raise ValueError("Indices of X_df and encoded_df have to match.")
    X_df_array = X_df.values
    n_atoms = np.sum(encoded_df.values, axis=1, keepdims=True)
    X_df_norm = X_df_array / n_atoms
    X_df_norm = pd.DataFrame(X_df_norm, index=X_df.index, columns=X_df.columns)
    X_df_norm.to_csv(save_path)


# Encode compound in terms of composition.
def composition_encoding(atomic_df, species, compound_name):
    encoded_df = pd.DataFrame(np.array([0]*len(atomic_df.index)).reshape((1,-1)),
                        columns=atomic_df.index)
    for element in species:
        encoded_df[str(element)] += 1
    encoded_df.index = [compound_name]

    return encoded_df


# Compute elemental average descriptors.
def compute_ave_descriptors(atomic_df, encoded_df, save_path):
    atomic_df = atomic_df.fillna(0) # fill the missing values with zero
    atomic_df_array = atomic_df.values
    encoded_df_array = encoded_df.values
    sum_rep = np.matmul(encoded_df_array, atomic_df_array)
    n_atoms = np.sum(encoded_df_array, axis=1, keepdims=True)
    ave_descriptors = sum_rep / n_atoms
    ave_descriptors = pd.DataFrame(ave_descriptors,
                                index=encoded_df.index, columns=atomic_df.columns)
    ave_descriptors.to_csv(save_path)


# Compute elemental std descriptors.
def compute_std_descriptors(atomic_df, ave_data, encoded_df, save_path):
    atomic_df = atomic_df.fillna(0) # fill the missing values with zero
    atomic_df_array = atomic_df.values
    ave_data = ave_data.fillna(0)
    ave_data_array = ave_data.values
    encoded_df_array = encoded_df.values
    assert len(ave_data_array) == len(encoded_df_array)

    std_descriptors = []
    for i in range(len(ave_data_array)):
        variance = (abs(atomic_df_array - ave_data_array[i]))**2
        weighted_variance_sum = np.matmul(encoded_df_array[i], variance)
        weighted_std = np.sqrt(weighted_variance_sum/(np.sum(encoded_df_array[i])))
        std_descriptors.append(weighted_std)

    std_descriptors = pd.DataFrame(std_descriptors, index=encoded_df.index,
                                    columns=atomic_df.columns)
    std_descriptors.to_csv(save_path)
