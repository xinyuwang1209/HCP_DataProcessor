__auther__ = 'Xinyu Wang'


import os
import pathos
import numpy as np
import pandas as pd
import multiprocessing
import math

from ._correlation_matrix_collector import correlation_matrix_collector


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

    grmid = pd.read_csv('/shared/healthinfolab/hcpdata/filted_HCPgrm.grm.id',sep=' ',header=None)
    grmid[1] = grmid[1].astype(int)
    data = data.loc[data['Subject'].isin(grmid[1])]
    return data

def get_X_y(data,use_4_features=False,normalize=True):
    X = data.iloc[:,1:-1]
    y = data.iloc[:,-1]
    if normalize:
        X = (X - X.mean()) / X.std()
    # Use 71,72,73,74
    if use_4_features:
        select_features_ids = [70,71,72,73]
        columns = []
        for i in range(X.shape[1]):
            exists = False
            for id in select_features_ids:
                if id in X.columns[i]:
                    exists = True
            if exists:
                columns.append(X.columns[i])
        X = X[columns]
    X.reset_index(drop=True,inplace=True)
    y.reset_index(drop=True,inplace=True)
    return X,y

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_classification
# from sklearn.model_selection import cross_val_score
#
# # Cross validation
# pool = []
# errors = []
# best = 1
# best_id = 0
# n = 5
# for i in range(n):
#     # X_train = X.iloc[i,:]
#     # y_test = X.iloc
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
#     X_train.reset_index(drop=True,inplace=True)
#     y_train.reset_index(drop=True,inplace=True)
#     X_test.reset_index(drop=True,inplace=True)
#     y_test.reset_index(drop=True,inplace=True)
#     clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0).fit(X_train, y_train)
#     # clf = svm.SVC(kernel='linear', C=1,degree=1,tol=0.001).fit(X_train, y_train)
#     pool.append(clf)
#     # if y_test[0] == 0:
#     #     y_test[0] = 1
#     # else:
#     #     y_test[0] = 0
#     #
#     result = pd.DataFrame(clf.predict(X_test))
#     result = result[0]
#     error = math.sqrt(((result - y_test)**2).sum(axis=0)/len(y_test))
#     errors.append(error)
#     if abs(error-0.5) > abs(best-0.5):
#         best_id = i
#         best = error
#
# clf = pool[best_id]
#
#
#
# clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
# scores = cross_val_score(clf, X, y, cv=5)
#

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

def get_10_features(feat,verbose=False):
    # Get best 10 entries
    coef_ = abs(feat)
    features_index = [i[0] for i in sorted(enumerate(coef_), key=lambda x:x[1])][:10]
    best_10_features = data.iloc[:,[0]+[i+1 for i in features_index]]
    # best_10_features.to_csv('/shared/healthinfolab/hcpdata/aal_corr_matrices/best_10_features.csv')
    if not verbose:
        return best_10_features
    else:
        feature_description = pd.read_excel('/shared/healthinfolab/hcpdata/aal_corr_matrices/aal_roi_list.xlsx')
        top_10_features_description = []
        for i in range(1,best_10_features.shape[1]):
            a, b = best_10_features.columns[i]
            a, b = a + 1, b + 1
            top_10_features_description.append(set(feature_description.loc[feature_description['AAL_no.'].isin([a,b])]['ROI']))
    return top_10_features_description

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
