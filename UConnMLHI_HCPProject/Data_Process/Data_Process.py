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

def correlation_collector(path_directory):
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
