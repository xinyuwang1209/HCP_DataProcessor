import numpy as np
import pandas as pd
# from Prediction_Utils.Age_Preddiction.Use_Lasso import run_lasso_cv
# from Prediction_Utils.Age_Preddiction.Use_SVR   import run_svm_cv
# from Prediction_Utils.Age_Preddiction.Use_RF    import run_rf_cv

#
#
# feat = run_svm_cv(X,y,kernel='linear',C=0.5,cv=5,degree=1)
# feat = run_svm_cv(X,y,kernel='rbf',C=0.5,cv=5,degree=1)
#
# # 0.42
# feat = run_svm_cv(X,y,kernel='poly',C=0.5,cv=5,degree=1)
# # 0.54
# feat = run_svm_cv(X,y,kernel='poly',C=0.5,cv=5,degree=10,coef0=100)
# feat = run_svm_cv(X,y,kernel='poly',C=0.5,cv=5,degree=1)
import HCP_DataProcessor.Data_Process as HCPDPP


def get_data(path=None,use_4_features=False,pca_filter=0,quality_filter=True):
    data = HCPDPP.correlation_matrix_collector()
    data = HCPDPP.update_addiction(data)
    data = HCPDPP.filter_has_genotype(data)
    if quality_filter:
        data = HCPDPP.filter_quality_test(data)
    X,y = HCPDPP.get_X_y(data,use_4_features=use_4_features)
    if pca_filter > 0:
        X,a = HCPDPP.pca_filter(X,pca_filter)
        return X,y,a
    else:
        return X,y
