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



description_list = [['Precentral_L','Left precentral gyrus','the area in the left parietal lobe posterior to the central sulcus'],
 ['Precentral_R','Right precentral gyrus','the gyrus in the right hemisphere anterior to the central sulcus. It is the primary motor area for the left part of the body'],
 ['Frontal_Sup_L','Left superior frontal gyrus','the left hemisphere part of the superior frontal gyrus'],
 ['Frontal_Sup_R','Right superior frontal gyrus','right hemisphere part of the superior frontal gyrus'],
 ['Frontal_Sup_Orb_L','Left superior frontal gyrus, orbital part',''],
 ['Frontal_Sup_Orb_R','Right superior frontal gyrus, orbital part',''],
 ['Frontal_Mid_L','Left middle frontal gyrus',''],
 ['Frontal_Mid_R','Right middle frontal gyrus',''],
 ['Frontal_Mid_Orb_L','Left middle frontal gyrus, orbital part',''],
 ['Frontal_Mid_Orb_R','Right middle frontal gyrus, orbital part',''],
 ['Frontal_Inf_Oper_L','Left inferior frontal gyrus, pars opercularis','the left hemisphere part of the inferior frontal gyrus, pars opercularis'],
 ['Frontal_Inf_Oper_R','Right inferior frontal gyrus, pars opercularis','the right hemisphere part of the inferior frontal gyrus, pars opercularis'],
 ['Frontal_Inf_Tri_L','Left inferior frontal gyrus, pars triangularis','the left hemisphere part of the inferior frontal gyrus, pars triangularis'],
 ['Frontal_Inf_Tri_R','Right inferior frontal gyrus, pars triangularis','the right hemisphere part of the inferior frontal gyrus, pars triangularis'],
 ['Frontal_Inf_Orb_L','Left inferior frontal gyrus, pars orbitalis',''],
 ['Frontal_Inf_Orb_R','Right inferior frontal gyrus, pars orbitalis',''],
 ['Rolandic_Oper_L','Left Rolandic operculum',''],
 ['Rolandic_Oper_R','Right Rolandic operculum',''],
 ['Supp_Motor_Area_L','Left supplementary motor area','the left hemisphere part of the brain region supplementary motor area'],
 ['Supp_Motor_Area_R','Right supplementary motor area','the right hemisphere part of the brain region supplementary motor area'],
 ['Olfactory_L','Left olfactory cortex',''],
 ['Olfactory_R','Right olfactory cortex',''],
 ['Frontal_Sup_Medial_L','Left medial frontal gyrus',''],
 ['Frontal_Sup_Medial_R','Right medial frontal gyrus',''],
 ['Frontal_Med_Orb_L','Left medial orbitofrontal cortex',''],
 ['Frontal_Med_Orb_R','Right medial orbitofrontal cortex',''],
 ['Rectus_L','Left gyrus rectus','the right hemisphere part of the gyrus rectus in the inferior medial frontal lobe'],
 ['Rectus_R','Right gyrus rectus','the right hemisphere part of the gyrus rectus in the inferior medial frontal lobe'],
 ['Insula_L','Left insula','a brain region in the cerebral cortex of the right hemisphere'],
 ['Insula_R','Right insula','a brain region in the cerebral cortex of the right hemisphere'],
 ['Cingulum_Ant_L','Left anterior cingulate gyrus','the left hemisphere part of the anterior cingulate gyrus'],
 ['Cingulum_Ant_R','Right anterior cingulate gyrus','the right hemisphere part of the anterior cingulate gyrus'],
 ['Cingulum_Mid_L','Left midcingulate area',''],
 ['Cingulum_Mid_R','Right midcingulate area',''],
 ['Cingulum_Post_L','Left posterior cingulate gyrus','a brain region in the left side of the back part of the cingulate gyrus'],
 ['Cingulum_Post_R','Right posterior cingulate gyrus','a brain region in the right side of the back part of the cingulate gyrus'],
 ['Hippocampus_L','Left hippocampus','the left brain part of the hippocampus in the medial temporal lobe'],
 ['Hippocampus_R','Right hippocampus','the right hemisphere part of the hippocampus'],
 ['ParaHippocampal_L','Left parahippocampal gyrus',''],
 ['ParaHippocampal_R','Right parahippocampal gyrus',''],
 ['Amygdala_L','Left amygdala','the left hemisphere part of the amygdala'],
 ['Amygdala_R','Right amygdala','the right hemisphere part of the amygdala brain region'],
 ['Calcarine_L','Left calcarine sulcus',''],
 ['Calcarine_R','Right calcarine sulcus',''],
 ['Cuneus_L','Left cuneus','the left hemisphere part of the cuneus in the occipital lobe'],
 ['Cuneus_R','Right cuneus','the right hemispere part of the cuneus brain region in the occipital lobe'],
 ['Lingual_L','Left lingual gyrus','the left hemisphere part of the lingual gyrus'],
 ['Lingual_R','Right lingual gyrus','the right hemisphere part of the lingual gyrus'],
 ['Occipital_Sup_L','Left superior occipital',''],
 ['Occipital_Sup_R','Right superior occipital',''],
 ['Occipital_Mid_L','Left middle occipital gyrus',''],
 ['Occipital_Mid_R','Right middle occipital gyrus',''],
 ['Occipital_Inf_L','Left inferior occipital cortex',''],
 ['Occipital_Inf_R','Right inferior occipital cortex',''],
 ['Fusiform_L','Left fusiform gyrus',''],
 ['Fusiform_R','Left fusiform gyrus',''],
 ['Postcentral_L','Left postcentral gyrus','the left hemisphere part of the postcentral gyrus. It is involved in somatosensory processing from the right side of the body'],
 ['Postcentral_R','Right postcentral gyrus',''],
 ['Parietal_Sup_L','Left superior parietal lobule',''],
 ['Parietal_Sup_R','Right superior parietal lobule','the upper part of the right parietal lobe'],
 ['Parietal_Inf_L','Left inferior parietal lobule',''],
 ['Parietal_Inf_R','Right inferior parietal lobule','the right hemisphere part of the inferior parietal lobule'],
 ['SupraMarginal_L','Left supramarginal gyrus',''],
 ['SupraMarginal_R','Right supramarginal gyrus',''],
 ['Angular_L','Left angular gyrus',''],
 ['Angular_R','Right angular gyrus',''],
 ['Precuneus_L','Left precuneus','the brain region "precuneus" in the left hemisphere'],
 ['Precuneus_R','Right precuneus','a brain region in the posterior part of the brain, posterior to the posterior cingulate'],
 ['Paracentral_Lobule_L','Left paracentral lobule',''],
 ['Paracentral_Lobule_R','Right paracentral lobule',''],
 ['Caudate_L','Left caudate nucleus','a basal ganglia brain region in the left hemisphere'],
 ['Caudate_R','Right caudate nucleus','the right hemisphere part of the caudate nucleus'],
 ['Putamen_L','Left putamen','the left hemisphere part of the basal ganglia brain region putamen'],
 ['Putamen_R','Right putamen','the basal ganglia structure in the right hemisphere. '],
 ['Pallidum_L','Left globus pallidus','the left hemisphere part of the globus pallidus'],
 ['Pallidum_R','Right globus pallidus',''],
 ['Thalamus_L','Left thalamus','the left brain part of the thalamus'],
 ['Thalamus_R','Right thalamus',''],
 ['Heschl_L','Left transverse temporal gyrus',''],
 ['Heschl_R','Right transverse temporal gyrus',''],
 ['Temporal_Sup_L','Left superior temporal gyrus',''],
 ['Temporal_Sup_R','Right superior temporal gyrus',''],
 ['Temporal_Pole_Sup_L','Left superior temporal pole','the left hemisphere part of the superior portion of the temporal pole'],
 ['Temporal_Pole_Sup_R','Right superior temporal pole','the right hemisphere part of the superior portion of the temporal pole'],
 ['Temporal_Mid_L','Left middle temporal gyrus','a gyrus in the left temporal lobe between the superior temporal gyrus and the inferior temporal gyrus'],
 ['Temporal_Mid_R','Right middle temporal gyrus','the right hemisphere part of the middle temporal gyrus'],
 ['Temporal_Pole_Mid_L','Left middle temporal pole',''],
 ['Temporal_Pole_Mid_R','Right middle temporal pole',''],
 ['Temporal_Inf_L','Left inferior temporal gyrus',''],
 ['Temporal_Inf_R','Right inferior temporal gyrus','the right hemisphere part of the inferior temporal gyrus in the temporal lobe'],
 ['Cerebelum_Crus1_L','Left crus I of cerebellar hemisphere',''],
 ['Cerebelum_Crus1_R','Right crus I of cerebellar hemisphere',''],
 ['Cerebelum_Crus2_L','Left crus II of cerebellar hemisphere',''],
 ['Cerebelum_Crus2_R','Right crus II of cerebellar hemisphere',''],
 ['Cerebelum_3_L','Left lobule III of cerebellar hemisphere',''],
 ['Cerebelum_3_R','Right lobule III of cerebellar hemisphere',''],
 ['Cerebelum_4_5_L','Left lobule IV, V of cerebellar hemisphere',''],
 ['Cerebelum_4_5_R','Right lobule IV, V of cerebellar hemisphere',''],
 ['Cerebelum_6_L','Left lobule VI of cerebellar hemisphere',''],
 ['Cerebelum_6_R','Right lobule VI of cerebellar hemisphere',''],
 ['Cerebelum_7b_L','Left lobule VIIB of cerebellar hemisphere',''],
 ['Cerebelum_7b_R','Right lobule VIIB of cerebellar hemisphere',''],
 ['Cerebelum_8_L','Left lobule VIII of cerebellar hemisphere',''],
 ['Cerebelum_8_R','Right lobule VIII of cerebellar hemisphere',''],
 ['Cerebelum_9_L','Left lobule IX of cerebellar hemisphere',''],
 ['Cerebelum_9_R','Right lobule IX of cerebellar hemisphere',''],
 ['Cerebelum_10_L','Left lobule X of cerebellar hemisphere',''],
 ['Cerebelum_10_R','Left lobule X of cerebellar hemisphere',''],
 ['Vermis_1_2','Lobule I, II of vermis',''],
 ['Vermis_3','Lobule III of vermis',''],
 ['Vermis_4_5','Lobule IV, V of vermis',''],
 ['Vermis_6','Lobule VI of vermis',''],
 ['Vermis_7','Lobule VII of vermis',''],
 ['Vermis_8','Lobule VIII of vermis',''],
 ['Vermis_9','Lobule IX of vermis',''],
 ['Vermis_10','Lobule X of vermis','']]




for region in description_list:
    feature_description.loc[feature_description['ROI'].str.contains(region[0]),'feature_description'] = region[1]
    feature_description.loc[feature_description['ROI'].str.contains(region[0]),'details'] = region[2]
    feature_description.loc[feature_description['ROI'].str.contains(region[0]),'ROI'] = region[0]





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



def get_description(best_10_features):
    feature_description = pd.read_csv('feature_description_details.csv')
    top_10_features_description = pd.DataFrame(columns=['region1','region2','region1_description','region2_description','region1_details','region2_details'])
    for i in range(1,best_10_features.shape[1]):
        a, b = best_10_features.columns[i][1:-1].split(', ')
        a, b = int(a) + 1, int(b) + 1
        a, b = feature_description.loc[feature_description['AAL_no.'].isin([a,b])]['ROI']
        region1_description, region1_details = feature_description.loc[feature_description['ROI'].str.contains(a)][['feature_description','details']].T.iloc[:,0]
        region2_description, region2_details = feature_description.loc[feature_description['ROI'].str.contains(b)][['feature_description','details']].T.iloc[:,0]
        top_10_features_description.loc[i] = {'region1': a,
                                              'region2': b,
                                              'region1_description': region1_description,
                                              'region2_description': region2_description,
                                              'region1_details': region1_details,
                                              'region2_details': region2_details}
    return top_10_features_description



description_female = get_description(best_10_features_female)
description_female[['region1_description','region2_description']]
description_male = get_description(best_10_features_male)
description_male[['region1_description','region2_description']]


def get_10_features(clf,data,gender=True,n=10):
    if clf.coef_.shape[0] == 1:
        feat = clf.coef_[0]
    else:
        feat = clf.coef_
    features_index = feat.argsort()[::-1]
    if gender:
        feature_size = (data.shape[1]-1)//2
        best_10_index_male = [i for i in features_index if i < feature_size]
        best_10_features_male = data.iloc[:,[0]+[i+1 for i in best_10_index_male[:n]]]
        # append columns for female calculation
        feat = feat[:feature_size] + feat[feature_size:]
        best_10_index_female = feat.argsort()[::-1]
        best_10_features_female = data.iloc[:,[0]+[i+1 for i in best_10_index_female[:n]]]
        return best_10_features_male,best_10_features_female
    else:
        best_10_features = data.iloc[:,[0]+[i+1 for i in features_index[:n]]]
        return best_10_features


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
