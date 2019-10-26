__auther__ = 'Xinyu Wang'


import os
import pathos
import numpy as np
import pandas as pd
import multiprocessing
import math


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






def addiction_reader(path_file='/shared/healthinfolab/hcpdata/aal_corr_matrices/number_addiction/',Dp=True,Ab=True):
    df = pd.read_csv(path_file + "psychiatric_data_HCP.csv")
    if Dp and Ab:
        alcohol_addiction = df.loc[(df['SSAGA_Alc_D4_Ab_Sx']>=1)
                                  |(df['SSAGA_Alc_D4_Dp_Sx']>=1)]
    elif Dp:
        alcohol_addiction = df.loc[(df['SSAGA_Alc_D4_Dp_Sx']>=1)]
    elif Ab:
        alcohol_addiction = df.loc[(df['SSAGA_Alc_D4_Ab_Sx']>=1)]
    else:
        alcohol_addiction = df.loc[(df['SSAGA_Alc_D4_Ab_Sx']>=1)
                                  &(df['SSAGA_Alc_D4_Dp_Sx']>=1)]
    return alcohol_addiction


# Read correlation matrix
# data = correlation_matrix_collector()

# Update addiction
def update_addiction(data,**args):
    data['addiction'] = 0
    addiction = addiction_reader(**args)
    data.loc[data['Subject'].isin(addiction['Subject']),'addiction'] = 1
    return data

# Update has genotype
def filter_has_genotype(data):
    path_file='/shared/healthinfolab/hcpdata/aal_corr_matrices/number_addiction/'
    geno = pd.read_csv(path_file + "psychiatric_data_HCP.csv")
    geno = geno.loc[geno["HasGT"]==True]
    data = data.loc[data['Subject'].isin(geno['Subject'])]
    return data

# Update quality
def filter_quality_test(data):
    subject_list = pd.read_csv('/shared/healthinfolab/hcpdata/aal_corr_matrices/result.txt',sep=' ',header=None)
    data = data.loc[data['Subject'].isin(subject_list[0])]
    data.drop_duplicates(subset=['Subject'],keep='last',inplace=True)
#     grmid = pd.read_csv('/shared/healthinfolab/hcpdata/filted_HCPgrm.grm.id',sep=' ',header=None)
#     grmid[1] = grmid[1].astype(int)
#     data = data.loc[data['Subject'].isin(grmid[1])]
    return data



from scipy.stats import spearmanr
n_clusters=3
n_components=20
alpha = 0.005
SUBJECT = 'Subject'
func_conn = filter_quality_test(filter_has_genotype(correlation_matrix_collector()))
psychiatric = pd.read_csv('/shared/healthinfolab/hcpdata/aal_corr_matrices/psychiatric_data_HCP.csv')
func_conn = func_conn.loc[func_conn[SUBJECT].isin(psychiatric[SUBJECT])]
psychiatric = psychiatric.loc[psychiatric[SUBJECT].isin(func_conn[SUBJECT])]
func_conn.sort_values(by=[SUBJECT],inplace=True)
psychiatric.sort_values(by=[SUBJECT],inplace=True)
func_conn.reset_index(drop=True,inplace=True)
psychiatric.reset_index(drop=True,inplace=True)

alcohol_features = ['SSAGA_Alc_D4_Dp_Sx',
                    'SSAGA_Alc_D4_Ab_Dx',
                    'SSAGA_Alc_D4_Ab_Sx',
                    'SSAGA_Alc_D4_Dp_Dx',
                    'SSAGA_Alc_12_Drinks_Per_Day',
                    'SSAGA_Alc_12_Frq',
                    'SSAGA_Alc_12_Frq_5plus',
                    'SSAGA_Alc_12_Frq_Drk',
                    'SSAGA_Alc_12_Max_Drinks',
                    'SSAGA_Alc_Age_1st_Use',
                    'SSAGA_Alc_Hvy_Drinks_Per_Day',
                    'SSAGA_Alc_Hvy_Frq',
                    'SSAGA_Alc_Hvy_Frq_5plus',
                    'SSAGA_Alc_Hvy_Frq_Drk',
                    'SSAGA_Alc_Hvy_Max_Drinks']

tobacco_features = ['SSAGA_FTND_Score',
                    'SSAGA_HSI_Score',
                    'SSAGA_TB_Age_1st_Cig',
                    'SSAGA_TB_DSM_Difficulty_Quitting',
                    'SSAGA_TB_DSM_Tolerance',
                    'SSAGA_TB_DSM_Withdrawal',
                    'SSAGA_TB_Hvy_CPD',
                    'SSAGA_TB_Max_Cigs',
                    'SSAGA_TB_Reg_CPD',
                    'SSAGA_TB_Smoking_History',
                    'SSAGA_TB_Still_Smoking',
                    'SSAGA_TB_Yrs_Since_Quit',
                    'SSAGA_TB_Yrs_Smoked']


alcohol_columns = [SUBJECT]
tobacco_columns = [SUBJECT]
log = []

for i in range(len(func_conn.columns[1:]:
    column = func_conn.columns[i]
    for alcohol_feature in alcohol_features:
        if alcohol_feature in psychiatric.columns:
            coef, p = spearmanr(func_conn[column], psychiatric[alcohol_feature], nan_policy='omit')
            if p <= alpha:
                alcohol_columns.append(column)
                alcohol_columns = list(set(alcohol_columns))
                log.append(['p value:',p, column, alcohol_feature])
                print(len(alcohol_columns)-1, 'p value:',p, coef, column, alcohol_feature)
    for tobacco_feature in tobacco_features:
        if tobacco_feature in psychiatric.columns:
            coef, p = spearmanr(func_conn[column], psychiatric[tobacco_feature], nan_policy='omit')
            if p <= alpha:
                tobacco_columns.append(column)
                tobacco_columns = list(set(tobacco_columns))
                log.append(['p value:',p, column, tobacco_feature])
                print(len(tobacco_columns)-1, 'p value:',p, coef, column, tobacco_feature)


func_conn_alcohol = func_conn[alcohol_columns]
func_conn_tobacco = func_conn[tobacco_columns]







# ## PCA and IPCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components=n_components)
# func_conn_alcohol_pca = pca.fit_transform(func_conn_alcohol)
#
# pca = PCA(n_components=n_components)
# func_conn_tobacco_pca = pca.fit_transform(func_conn_tobacco)
#
# from sklearn.decomposition import IncrementalPCA
# transformer = IncrementalPCA(n_components=n_components, batch_size=200)
# func_conn_alcohol_ipca = transformer.fit_transform(func_conn_alcohol)
#
# transformer = IncrementalPCA(n_components=n_components, batch_size=200)
# func_conn_tobacco_ipca = transformer.fit_transform(func_conn_tobacco)



# Load label
alcohol_labels = pd.read_csv('/shared/healthinfolab/hcpdata/aal_corr_matrices/full_alc_label.csv')
tobacco_labels = pd.read_csv('/shared/healthinfolab/hcpdata/aal_corr_matrices/full_tob_label.csv')

alcohol_labels = alcohol_labels.loc[alcohol_labels[SUBJECT].isin(func_conn[SUBJECT])]
tobacco_labels = tobacco_labels.loc[tobacco_labels[SUBJECT].isin(func_conn[SUBJECT])]

alcohol_labels.sort_values(by=[SUBJECT],inplace=True)
tobacco_labels.sort_values(by=[SUBJECT],inplace=True)
alcohol_labels.reset_index(drop=True,inplace=True)
tobacco_labels.reset_index(drop=True,inplace=True)

func_conn_alcohol = func_conn_alcohol.drop(['label','label_x','label_y'],axis=1,errors='ignore')
func_conn_tobacco = func_conn_tobacco.drop(['label','label_x','label_y'],axis=1,errors='ignore')
func_conn_alcohol = func_conn_alcohol.merge(alcohol_labels,on=SUBJECT)
func_conn_tobacco = func_conn_tobacco.merge(tobacco_labels,on=SUBJECT)
func_conn_alcohol = func_conn_alcohol.drop(['label_x','label_y'],axis=1,errors='ignore')
func_conn_tobacco = func_conn_tobacco.drop(['label_x','label_y'],axis=1,errors='ignore')

X_alcohol = func_conn_alcohol.drop(['label',SUBJECT,'addiction','label_x','label_y'],axis=1,errors='ignore')
y_alcohol = func_conn_alcohol['label']
X_tobacco = func_conn_tobacco.drop(['label',SUBJECT,'addiction','label_x','label_y'],axis=1,errors='ignore')
y_tobacco = func_conn_tobacco['label']


import numpy as np
import pandas as pd
from sklearn import metrics
import random
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import KFold
from scipy.stats import norm
selectedfeatureNumber=20
number=len(Columns)
Num_iteration=50
avg_variable=None
for i in range(Num_iteration):
    seed=random.randint(1,10000)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    clf = LogisticRegression(penalty='l2',C=5,random_state=seed)
    clf.fit(X_alcohol, y_alcohol)
    #print(clf.score(X_train, y_train))
    #print(sum(y_test)/len(y_test))
    #print(clf.score(X_test, y_test))
    para=clf.coef_.tolist()
    para=para[0]
    if(avg_variable is None):
        avg_variable=para
    else:
        avg_variable = [avg_variable[i]+para[i] for i in range(len(para))]

top_n_idx = np.argsort([abs(a) for a in avg_variable])[::-1][:selectedfeatureNumber]

func_conn_alcohol_20 = func_conn_alcohol[[SUBJECT] + list(X_alcohol.iloc[:,top_n_idx].columns) + ['label']]
X_alcohol_20 = func_conn_alcohol[X_alcohol.iloc[:,top_n_idx].columns]




for i in range(Num_iteration):
    seed=random.randint(1,10000)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    clf = LogisticRegression(penalty='l2',C=5,random_state=seed)
    clf.fit(X_tobacco, y_tobacco)
    #print(clf.score(X_train, y_train))
    #print(sum(y_test)/len(y_test))
    #print(clf.score(X_test, y_test))
    para=clf.coef_.tolist()
    para=para[0]
    if(avg_variable is None):
        avg_variable=para
    else:
        avg_variable = [avg_variable[i]+para[i] for i in range(len(para))]

top_n_idx = np.argsort([abs(a) for a in avg_variable])[::-1][:selectedfeatureNumber]

func_conn_tobacco_20 = func_conn_tobacco[[SUBJECT] + list(X_tobacco.iloc[:,top_n_idx].columns) + ['label']]
X_tobacco_20 = func_conn_tobacco[X_tobacco.iloc[:,top_n_idx].columns]


# K Mean Clustering
from sklearn.cluster import KMeans
kmeans_alcohol = KMeans(n_clusters=n_clusters, random_state=0).fit(X_alcohol_20)
kmeans_alcohol.labels_

kmeans_tobacco = KMeans(n_clusters=n_clusters, random_state=0).fit(X_tobacco_20)
kmeans_tobacco.labels_

#
#
# from sklearn.cluster import KMeans
# kmeans_alcohol = KMeans(n_clusters=n_clusters, random_state=0).fit(func_conn_alcohol_20.drop(['label',SUBJECT,'addiction','label_x','label_y'],axis=1,errors='ignore'))
# kmeans_alcohol.labels_
#
# kmeans_tobacco = KMeans(n_clusters=n_clusters, random_state=0).fit(func_conn_tobacco_20.drop(['label',SUBJECT,'addiction','label_x','label_y'],axis=1,errors='ignore'))
# kmeans_tobacco.labels_
#
import pickle
pickle.dump(kmeans_alcohol.labels_, open('/shared/healthinfolab/hcpdata/aal_corr_matrices/pickle_kmeans_alcohol_labels.pickle','wb'))
pickle.dump(kmeans_tobacco.labels_, open('/shared/healthinfolab/hcpdata/aal_corr_matrices/pickle_kmeans_tobacco_labels.pickle','wb'))
pickle.dump(func_conn_alcohol_20, open('/shared/healthinfolab/hcpdata/aal_corr_matrices/pickle_func_conn_alcohol_20.pickle','wb'))
pickle.dump(func_conn_tobacco_20, open('/shared/healthinfolab/hcpdata/aal_corr_matrices/pickle_func_conn_tobacco_20.pickle','wb'))
















scp xiw14035@login.storrs.hpc.uconn.edu:/shared/healthinfolab/hcpdata/aal_corr_matrices/pickle_kmeans_alcohol_labels.pickle /home/xinyu/src/HCP/pickle_kmeans_alcohol_labels.pickle
scp xiw14035@login.storrs.hpc.uconn.edu:/shared/healthinfolab/hcpdata/aal_corr_matrices/pickle_kmeans_tobacco_labels.pickle /home/xinyu/src/HCP/pickle_kmeans_tobacco_labels.pickle
scp xiw14035@login.storrs.hpc.uconn.edu:/shared/healthinfolab/hcpdata/aal_corr_matrices/pickle_func_conn_alcohol_20.pickle /home/xinyu/src/HCP/pickle_func_conn_alcohol_20.pickle
scp xiw14035@login.storrs.hpc.uconn.edu:/shared/healthinfolab/hcpdata/aal_corr_matrices/pickle_func_conn_tobacco_20.pickle /home/xinyu/src/HCP/pickle_func_conn_tobacco_20.pickle


import numpy as np
import pandas as pd
import pickle


home_path = '/home/xinyu/src/'
kmeans_alcohol_labels = pickle.load(open(home_path + 'HCP/pickle_kmeans_alcohol_labels.pickle','rb'))
kmeans_tobacco_labels = pickle.load(open(home_path + 'HCP/pickle_kmeans_tobacco_labels.pickle','rb'))
func_conn_alcohol_20 = pickle.load(open(home_path + 'HCP/pickle_func_conn_alcohol_20.pickle','rb'))
func_conn_tobacco_20 = pickle.load(open(home_path + 'HCP/pickle_func_conn_tobacco_20.pickle','rb'))

import matplotlib.pyplot as plt
for i in range(20):
    plt.scatter(func_conn_alcohol_20.iloc[:,i+1], kmeans_alcohol_labels,c=kmeans_alcohol_labels)
    plt.show()



# use random forest on model 2
