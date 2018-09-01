import numpy as np
import pandas as pd
from getdata_from_poscar import DataConstruction
from data_processing import calculate_descriptors, composition_encoding, normalization
from data_processing import compute_ave_descriptors, compute_std_descriptors
from atomic_rep_to_descriptors import atomic_rep_to_compound_descriptor

# Setting path------------------------------------------------------------------
# dir_path = "/Users/keishiro/Documents/M2_research" # lab's laptop
dir_path = "/Users/user/Documents/M2_research" # my macbook

descriptors_dir = dir_path + "/data/to_kanamori/cohesive/descriptors/"
# compounds_list_dir = dir_path + "/data/to_kanamori/cohesive/compounds_name" # for cohesive energy data
compounds_list_dir = dir_path + "/data/to_kanamori/ltc/kappa" # for ltc data
# ------------------------------------------------------------------------------

# encoded_compounds = pd.read_csv(dir_path + "/data/seko/coh_energy/encoded_compounds.csv", index_col=0)
encoded_compounds = pd.read_csv(dir_path + "/data/seko/ltc/encoded_compounds.csv", index_col=0)

average_data = pd.read_csv(dir_path + "/data/seko/ltc/X_element_ave.csv", index_col=0)

X_element_ave = pd.read_csv(dir_path + "/data/seko/ltc/X_element_ave.csv", index_col=0)
X_element_std = pd.read_csv(dir_path + "/data/seko/ltc/X_element_std.csv", index_col=0)
X_nonperiodic_dot = pd.read_csv(dir_path + "/data/seko/ltc/X_nonperiodic_dot.csv", index_col=0)
X_nonperiodic_matmul = pd.read_csv(dir_path + "/data/seko/ltc/X_nonperiodic_matmul.csv", index_col=0)
X_periodic_dot = pd.read_csv(dir_path + "/data/seko/ltc/X_periodic_dot.csv", index_col=0)
X_periodic_matmul = pd.read_csv(dir_path + "/data/seko/ltc/X_periodic_matmul.csv", index_col=0)

atomic_data = pd.read_csv(dir_path + "/data/seko/atomic_data_20160603.csv", index_col=0)
atomic_data = atomic_data.drop(["Rps-d"], axis=1)
# atomic_data = atomic_data.drop(["Rps-d", "Cp-g", "Cp-mol"], axis=1)

def add_targetVariables(df, y_name, y_data):
    df[y_name] = y_data
    df.to_csv(dir_path + "/data/seko/X2016_y_data.csv")


# make descriptors by using Seko's code
def get_mean_std_cov(descriptor_path):
    with open(descriptor_path) as f:
        d = []
        lines = f.readlines()
        for i in range(len(lines)):
            d.append(list(map(float, lines[i].strip().split())))
        d = np.array(d)

        return atomic_rep_to_compound_descriptor(d)


if __name__ == "__main__":
    descriptors = []
    compounds = []
    mean_std = []

    # #set empty DataFrame for encoding
    # encoded = pd.DataFrame(index=[], columns=atomic_data.index)

    with open(compounds_list_dir) as f:
        lines = f.readlines()
        # for i in range(len(lines)):
        for i in range(1): # This line is just for test
            if i % 500 == 0:
                print(str(i) + "compounds are done!")
            # for cohesive energy data
            # compound_dir = lines[i].strip()
            # for ltc data
            line = lines[i].strip().split()
            compound_dir = line[0]
            compounds.append(compound_dir)

            # make descriptors, which combine elemental and structual information
            # POSCAR_path = descriptors_dir + compound_dir + "/POSCAR" # for cohesive energy data
            POSCAR_path = dir_path + compound_dir + "/POSCAR" # for ltc data
            descriptor = set_descriptors(atomic_data, POSCAR_path, cutoff=10,
                                        matmul=False, element_wise=True,
                                        periodic=False, attribute="all", sum_type="sum")
            descriptors.append(descriptor)

            # # make atomic and structural descriptors by Seko's code
            # descriptor_path = descriptors_dir + compound_dir + "/descriptors_st_cos40"
            # mean_std.append(get_mean_std_cov(descriptor_path))

            # # make encoded dataframe
            # dc = DataConstruction()
            # structure, n_atoms, species = dc.set_structure(POSCAR_path)
            # df = onehot_encoding(atomic_data, species, compound_dir)
            # encoded = encoded.append(df)

        # mean_std = np.array(mean_std)
        # # columns_list = ['BP', 'Cp-g', 'Cp-mol','E-fusion', 'E-vapor', 'EA', 'Group', 'IE1', 'IE2',\
        # #                 'kai(Allen)', 'kai(Pauling)', 'm', 'mol-vol', 'MP', 'Period', 'Ratom',\
        # #                 'Rcov', 'rho', 'Rps-d', 'Rps-p', 'Rps-s', 'Rvdw', 'Thermal-Cond', 'Z']
        # # df = pd.DataFrame(mean_std, index=compounds, columns=columns_list)
        # df = pd.DataFrame(mean_std, index=compounds)
        # df.to_csv(dir_path + '/data/seko/coh_energy/X_cos40.csv')

    # # make mixtured descriptors
    # columns_list = ['Z_diff', 'Period_diff', 'Group_diff', 'm_diff', 'kai(Pauling)_diff',\
    #                 'kai(Allen)_diff', 'EA_diff', 'IE1_diff', 'IE2_diff', 'Rps-s_diff', \
    #                 'Rps-p_diff', 'Rvdw_diff', 'Rcov_diff', 'MP_diff', 'BP_diff', 'Cp-g_diff',\
    #                 'Cp-mol_diff', 'rho_diff', 'E-fusion_diff','E-vapor_diff', 'Thermal-Cond_diff', \
    #                 'Ratom_diff', 'Mol-Vol_diff', 'Z_ave', 'Period_ave', 'Group_ave', \
    #                 'm_ave', 'kai(Pauling)_ave', 'kai(Allen)_ave', 'EA_ave', 'IE1_ave', \
    #                 'IE2_ave', 'Rps-s_ave', 'Rps-p_ave', 'Rvdw_ave', 'Rcov_ave', 'MP_ave', \
    #                 'BP_ave', 'Cp-g_ave', 'Cp-mol_ave', 'rho_ave', 'E-fusion_ave', 'E-vapor_ave', \
    #                 'Thermal-Cond_ave', 'Ratom_ave', 'Mol-Vol_ave', 'Z_std', 'Period_std', \
    #                 'Group_std', 'm_std', 'kai(Pauling)_std', 'kai(Allen)_std', 'EA_std', \
    #                 'IE1_std', 'IE2_std', 'Rps-s_std', 'Rps-p_std', 'Rvdw_std', 'Rcov_std', \
    #                 'MP_std', 'BP_std', 'Cp-g_std', 'Cp-mol_std', 'rho_std', 'E-fusion_std', \
    #                 'E-vapor_std', 'Thermal-Cond_std', 'Ratom_std', 'Mol-Vol_std']
    # df_descriptors = pd.DataFrame(np.array(descriptors),
    #                                 # if you set attribute to 'all' in set_descriptors function, use columns_list
    #                                 columns=columns_list,
    #                                 # columns=atomic_data.columns,
    #                                 index=compounds)
    # df_descriptors.to_csv(dir_path + "/data/seko/ltc/X_nonperiodic_matmul.csv")

    # # onehot encoding of compounds
    # encoded.to_csv(dir_path + "/data/seko/ltc/encoded_compounds.csv")

    # # make average descriptors
    # set_ave_descriptor(atomic_data, encoded_compounds, dir_path+'/data/seko/ltc/X_element_ave.csv')
    # # make std descriptors
    # set_std_descriptor(atomic_data, average_data, encoded_compounds, dir_path+'/data/seko/ltc/X_element_std.csv')

    # normalization(X_nonperiodic_dot, encoded_compounds, dir_path+"/data/seko/ltc/X_nonperiodic_dot_norm.csv")
    # normalization(X_nonperiodic_matmul, encoded_compounds, dir_path+"/data/seko/ltc/X_nonperiodic_matmul_norm.csv")
    # normalization(X_periodic_dot, encoded_compounds, dir_path+"/data/seko/ltc/X_periodic_dot_norm.csv")
    # normalization(X_periodic_matmul, encoded_compounds, dir_path+"/data/seko/ltc/X_periodic_matmul_norm.csv")
