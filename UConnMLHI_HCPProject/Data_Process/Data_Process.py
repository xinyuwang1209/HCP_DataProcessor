__auther__ = 'Xinyu Wang'

import os
import pathos
from scipy.io import loadmat
import numpy as np
import pandas as pd
import multiprocessing
import math
from sklearn import svm
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import train_test_split


def correlation_collector(path_directory='/shared/healthinfolab/hcpdata/aal_corr_matrices/aal_corr_matrices/',col2=False):
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
        result.append(mat_reader(file_path,col2=col2))
        print(i)
        i += 1
    return pd.DataFrame(result)

def mat_reader(path_file,col2=False):
    mat = loadmat(path_file)
    df = pd.DataFrame(mat[list(mat.keys())[-1]])
    del mat
    # Find indices of the upper triangular matrix
    keep = ~np.tril(np.ones(df.shape)).astype('bool')
    if col2:
        col_keep = [70,71,72,73]
        for i in range(len(keep)):
            if i not in col_keep:
                keep[i] = [False] * len(keep[i])
    keep = keep.reshape(df.size)
    # Convert it into series
    df = df.stack()[keep]
    # Get subject id
    id = int(path_file.split('/')[-1].split('.')[0].split('sess')[0])
    return pd.concat([pd.Series(id),df])
    # df = correlation_collector(os.getcwd())

def addiction_reader(path_file='/shared/healthinfolab/hcpdata/aal_corr_matrices/number_addiction/'):
    # subject_list = pd.read_csv('/shared/healthinfolab/hcpdata/aal_corr_matrices/result.txt',sep=' ',header=None)
    # data_compl=pd.read_csv(path_file + "HCP_summary_S1206.csv")
    data=pd.read_csv(path_file + "psychiatric_data_HCP.csv")
    # Full_MR_Compl=data_compl["3T_Full_MR_Compl"]
    # data_compl=data_compl[Full_MR_Compl==True]
    # data = data[data['Subject'].isin(data_compl["Subject"])]
    # data = data.loc[data['Subject'].isin(subject_list[0])]
    alcohol_addiction = data.loc[(data['SSAGA_Alc_D4_Dp_Sx']>=1)
                               | (data['SSAGA_Alc_D4_Ab_Sx']>=1)]
    return alcohol_addiction




data = correlation_collector(col2=False)
data.rename(columns={0:'Subject'},inplace=True)
data['Subject'] = data['Subject'].astype(int)
data['addiction'] = 0
addiction = addiction_reader()

path_file='/shared/healthinfolab/hcpdata/aal_corr_matrices/number_addiction/'
geno = pd.read_csv(path_file + "psychiatric_data_HCP.csv")
geno = geno.loc[geno["HasGT"]==True]
data = data.loc[data['Subject'].isin(geno['Subject'])]

data.loc[data['Subject'].isin(addiction['Subject']),'addiction'] = 1
subject_list = pd.read_csv('/shared/healthinfolab/hcpdata/aal_corr_matrices/result.txt',sep=' ',header=None)
data = data.loc[data['Subject'].isin(subject_list[0])]
data.drop_duplicates(subset=['Subject'],keep='last',inplace=True)

grmid = pd.read_csv('/shared/healthinfolab/hcpdata/filted_HCPgrm.grm.id',sep=' ',header=None)
grmid[1] = grmid[1].astype(int)
data = data.loc[data['Subject'].isin(grmid[1])]

from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split

from sklearn import svm

X = (X - X.mean()) / X.std()

# X = X - X.min()
# X = X / X.max()

X = data.iloc[:,1:-1]
X.reset_index(drop=True,inplace=True)
y = data.iloc[:,-1]
y.reset_index(drop=True,inplace=True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Cross validation
pool = []
errors = []
best = 1
best_id = 0
n = 5
for i in range(n):
    # X_train = X.iloc[i,:]
    # y_test = X.iloc
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    X_train.reset_index(drop=True,inplace=True)
    y_train.reset_index(drop=True,inplace=True)
    X_test.reset_index(drop=True,inplace=True)
    y_test.reset_index(drop=True,inplace=True)
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0).fit(X_train, y_test)
    # clf = svm.SVC(kernel='linear', C=1,degree=1,tol=0.001).fit(X_train, y_train)
    pool.append(clf)
    # if y_test[0] == 0:
    #     y_test[0] = 1
    # else:
    #     y_test[0] = 0
    #
    result = pd.DataFrame(clf.predict(X_test))
    result = result[0]
    error = math.sqrt(((result - y_test)**2).sum(axis=0)/len(y_test))
    errors.append(error)
    if abs(error-0.5) > abs(best-0.5):
        best_id = i
        best = error

clf = pool[best_id]

def feature_entry_locator(n):
    current = n
    n_column = 115
    row = 0
    while n_column <= current:
        current -= n_column
        n_column -= 1
        row += 1
    column = row + current + 1
    return row, column

# Get best 100 entries
features_index = [i[0] for i in sorted(enumerate(clf.coef_[0]), key=lambda x:x[1])][:10]
best_10_features = data.iloc[:,[0]+[i+1 for i in features_index]]
best_10_features.to_csv('/shared/healthinfolab/hcpdata/aal_corr_matrices/best_10_features.csv')






# use one session
# exclude some subject
# #
#
# Build model for alcohol
# 1. y list of alcohol featuress
#         alcohol abusee booelan
#         alcohol independence boolean
#
# TODO: use alcohol abusee
#     use 116x116 correlation features
#     model1 sparse_SVM!!!
#     model2 2*116 sparse_SVM
#
# Get top 10 featuress
#
# Tan run heritability
# narrow sense -h^2 heritability
#
#
# ?chip-BASED
# ?SOLAR
# TODO stratify
# double check value
# gaussian bandwith
# chose rbf kernel, print out the kernel
#

# if matrix is too small, then signma is too small
# rbf

# run random forest before svc, on 4 column things
#  run random forest is too expensieve on full corelation

# figure out feature selector
# feature selector



# use random forest on model 2
