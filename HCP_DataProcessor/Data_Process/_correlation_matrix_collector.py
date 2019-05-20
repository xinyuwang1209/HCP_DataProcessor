#!/bin/python3
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from queue import Queue
from threading import Thread
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

def correlation_matrix_collector(path_directory='/shared/healthinfolab/hcpdata/aal_corr_matrices/aal_corr_matrices/'):
    if not path_directory[-1] == '/':
        path_directory += '/'
    file_names = os.listdir(path_directory)
    file_names = [ path_directory + file_name for file_name in file_names]
    pool = ThreadPool(128)
    result = pool.map(mat_reader,file_names)
    pool.close()
    pool.join()
    df = pd.DataFrame(result)
    df.rename(columns={0:'Subject'},inplace=True)
    df['Subject'] = df['Subject'].astype(int)
    return df

def mat_reader(file_path):
    mat = loadmat(file_path)
    df = pd.DataFrame(mat[list(mat.keys())[-1]])
    del mat
    # Find indices of the upper triangular matrix
    keep = ~np.tril(np.ones(df.shape)).astype('bool')
    keep = keep.reshape(df.size)
    # Convert it into series
    df = df.stack()[keep]
    # Get subject id
    id = int(file_path.split('/')[-1].split('.')[0].split('sess')[0])
    return pd.concat([pd.Series(id),df])
