import numpy as np
import pandas as pd
from Prediction_Utils.Age_Preddiction.Use_Lasso import run_lasso_cv
from Prediction_Utils.Age_Preddiction.Use_SVR   import run_svm_cv
from Prediction_Utils.Age_Preddiction.Use_RF    import run_rf_cv

feat = run_svm_cv(X,y,kernel='linear',C=0.5,cv=5,degree=1)
feat = run_svm_cv(X,y,kernel='rbf',C=0.5,cv=5,degree=1)

# 0.42
feat = run_svm_cv(X,y,kernel='poly',C=0.5,cv=5,degree=1)
# 0.54
feat = run_svm_cv(X,y,kernel='poly',C=0.5,cv=5,degree=10,coef0=100)
feat = run_svm_cv(X,y,kernel='poly',C=0.5,cv=5,degree=1)
