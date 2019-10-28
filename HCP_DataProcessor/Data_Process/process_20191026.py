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

## kai_guan
kai_guan = 'alcohol'
# kai_guan = 'tobacco'

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
        addiction = df.loc[(df['SSAGA_Alc_D4_Ab_Sx']>=1)
                                  |(df['SSAGA_Alc_D4_Dp_Sx']>=1)]
    elif Dp:
        addiction = df.loc[(df['SSAGA_Alc_D4_Dp_Sx']>=1)]
    elif Ab:
        addiction = df.loc[(df['SSAGA_Alc_D4_Ab_Sx']>=1)]
    else:
        addiction = df.loc[(df['SSAGA_Alc_D4_Ab_Sx']>=1)
                                  &(df['SSAGA_Alc_D4_Dp_Sx']>=1)]
    return addiction


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
func_conn_raw = filter_quality_test(filter_has_genotype(correlation_matrix_collector()))
psychiatric = pd.read_csv('/shared/healthinfolab/hcpdata/aal_corr_matrices/psychiatric_data_HCP.csv')
func_conn = func_conn_raw.loc[func_conn_raw[SUBJECT].isin(psychiatric[SUBJECT])]
psychiatric = psychiatric.loc[psychiatric[SUBJECT].isin(func_conn[SUBJECT])]
func_conn.sort_values(by=[SUBJECT],inplace=True)
psychiatric.sort_values(by=[SUBJECT],inplace=True)
func_conn.reset_index(drop=True,inplace=True)
psychiatric.reset_index(drop=True,inplace=True)

features= dict()
features['alcohol'] = ['SSAGA_Alc_D4_Dp_Sx',
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

features['tobacco'] = ['SSAGA_FTND_Score',
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


columns = [SUBJECT]
log = []

for i in range(len(func_conn.columns[1:])):
    column = func_conn.columns[i]
    for feature in features[kai_guan]:
        if feature in psychiatric.columns:
            coef, p = spearmanr(func_conn[column], psychiatric[feature], nan_policy='omit')
            if p <= alpha:
                columns.append(column)
                columns = list(set(columns))
                log.append(['p value:',p, column, feature])
                print(len(columns)-1, 'p value:',p, coef, column, feature)


func_conn = func_conn[columns]







# ## PCA and IPCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components=n_components)
# func_conn_pca = pca.fit_transform(func_conn)
#
# pca = PCA(n_components=n_components)
# func_conn_tobacco_pca = pca.fit_transform(func_conn_tobacco)
#
# from sklearn.decomposition import IncrementalPCA
# transformer = IncrementalPCA(n_components=n_components, batch_size=200)
# func_conn_ipca = transformer.fit_transform(func_conn)
#
# transformer = IncrementalPCA(n_components=n_components, batch_size=200)
# func_conn_tobacco_ipca = transformer.fit_transform(func_conn_tobacco)



# Load label
if kai_guan == 'alcohol':
    file_name = 'full_alc_label.csv'
elif kai_guan == 'tobacco':
    file_name = 'full_tob_label.csv'

labels = pd.read_csv('/shared/healthinfolab/hcpdata/aal_corr_matrices/' + file_name)

labels = labels.loc[labels[SUBJECT].isin(func_conn[SUBJECT])]

labels.sort_values(by=[SUBJECT],inplace=True)
labels.reset_index(drop=True,inplace=True)

func_conn = func_conn.drop(['label','label_x','label_y'],axis=1,errors='ignore')
func_conn = func_conn.merge(labels,on=SUBJECT)
func_conn = func_conn.drop(['label_x','label_y'],axis=1,errors='ignore')

X = func_conn.drop(['label',SUBJECT,'addiction','label_x','label_y'],axis=1,errors='ignore')
y = func_conn['label']


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
    clf.fit(X, y)
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

func_conn_20 = func_conn[[SUBJECT] + list(X.iloc[:,top_n_idx].columns) + ['label']]
X_20 = func_conn[X.iloc[:,top_n_idx].columns]




# K Mean Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_20)
kmeans.labels_

#
#
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(func_conn_20.drop(['label',SUBJECT,'addiction','label_x','label_y'],axis=1,errors='ignore'))
# kmeans.labels_
#
# kmeans_tobacco = KMeans(n_clusters=n_clusters, random_state=0).fit(func_conn_tobacco_20.drop(['label',SUBJECT,'addiction','label_x','label_y'],axis=1,errors='ignore'))
# kmeans_tobacco.labels_
#
import pickle
pickle.dump(kmeans.labels_, open('/shared/healthinfolab/hcpdata/aal_corr_matrices/pickle_kmeans_' + kai_guan + '_labels.pickle','wb'))
pickle.dump(func_conn_20, open('/shared/healthinfolab/hcpdata/aal_corr_matrices/pickle_func_conn_' + kai_guan +'_20.pickle','wb'))


scp xiw14035@login.storrs.hpc.uconn.edu:/shared/healthinfolab/hcpdata/aal_corr_matrices/pickle_kmeans_labels.pickle /home/xinyu/src/HCP/pickle_kmeans_labels.pickle
scp xiw14035@login.storrs.hpc.uconn.edu:/shared/healthinfolab/hcpdata/aal_corr_matrices/pickle_func_conn_20.pickle /home/xinyu/src/HCP/pickle_func_conn_20.pickle


import numpy as np
import pandas as pd
import pickle


home_path = '/home/xinyu/src/'
kmeans_labels = pickle.load(open(home_path + 'HCP/pickle_kmeans_' + kai_guan + '_labels.pickle','rb'))
func_conn_20 = pickle.load(open(home_path + 'HCP/pickle_func_conn_' + kai_guan + '_20.pickle','rb'))

import matplotlib.pyplot as plt
for i in range(20):
    plt.scatter(func_conn_20.iloc[:,i+1], kmeans_labels,c=kmeans_labels)
    plt.show()



# use random forest on model 2
