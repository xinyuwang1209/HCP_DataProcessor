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

def get_data(path=None,use_4_features=False,pca_filter=0):
    data = .DataProcess.correlation_matrix_collector()
    data = .DataProcess.update_addiction(data)
    data = .DataProcess.filter_has_genotype(data)
    data = .DataProcess.filter_quality_test(data)
    X,y = .DataProcess.get_X_y(data,use_4_features=use_4_features)
    if pca_filter > 0:
        X,a = .DataProcess.pca_filter(X,pca_filter)
        return X,y,a
    else:
        return X,y
