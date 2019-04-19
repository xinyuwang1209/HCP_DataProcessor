__auther__ = 'Xinyu Wang'

import os
import pathos
from scipy.io import loadmat
import numpy as np
import pandas as pd
import multiprocessing

# # Import Utilities
# from ..Utilities.Utilities import *
#
# global progress
# progress = 0

def correlation_collector(path_directory='/shared/healthinfolab/hcpdata/aal_corr_matrices/aal_corr_matrices/'):
        end_in_slash = path_directory[-1] == '/'
        file_names = os.listdir(path_directory)
        list_file_paths = []
        for file_name in file_names:
            if end_in_slash:
                list_file_paths.append(path_directory + file_name)
            else:
                list_file_paths.append(path_directory + '/' + file_name)
        # print(list_file_paths)
        global file_size
        file_size = len(list_file_paths)
        print('file size:', file_size)
        # ncpu = multiprocessing.cpu_count()
        # pool = pathos.multiprocessing.ProcessingPool(ncpu).map
        # result = pool(mat_reader,list_file_paths)
        result = []
        i=0
        for file_path in list_file_paths:
            result.append(mat_reader(file_path))
            print(i)
            i += 1
        return pd.DataFrame(result)

def mat_reader(path_file):
    mat = loadmat(path_file)
    df = pd.DataFrame(mat[list(mat.keys())[-1]])
    del mat
    # Find indices of the upper triangular matrix
    keep = ~np.tril(np.ones(df.shape)).astype('bool').reshape(df.size)
    # Convert it into series
    df = df.stack()[keep]
    # Get subject id
    id = int(path_file.split('/')[-1].split('.')[0].split('sess')[0])
    return pd.concat([pd.Series(id),df])
    df = correlation_collector(os.getcwd())

def addiction_reader(path_file='/shared/healthinfolab/hcpdata/aal_corr_matrices/number_addiction/'):
    data_compl=pd.read_csv(path_file + "HCP_summary_S1206.csv")
    data=pd.read_csv(path_file + "psychiatric_data_HCP.csv")
    Full_MR_Compl=data_compl["3T_Full_MR_Compl"]
    data_compl=data_compl[Full_MR_Compl==True]
    data = data[data['Subject'].isin(data_compl["Subject"])]
    alcohol_addiction = data.loc[(data['SSAGA_Alc_D4_Dp_Sx']>=1)
                               | (data['SSAGA_Alc_D4_Ab_Sx']>=1)]
    return alcohol_addiction

data = correlation_collector()
data.rename(columns={0:'Subject'},inplace=True)
data['Subject'] = data['Subject'].astype(int)
data['addiction'] = 0
addiction = addiction_reader()
data.loc[data['Subject'].isin(addiction['Subject'],'addiction') = 1
