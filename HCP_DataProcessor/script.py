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


def get_data(path=None,use_4_features=False,pca_filter=0,quality_filter=True,normalize=True,**args):
    data = HCPDPP.correlation_matrix_collector()
    data = HCPDPP.update_addiction(data,**args)
    data = HCPDPP.filter_has_genotype(data)
    if quality_filter:
        data = HCPDPP.filter_quality_test(data)
    X,y = HCPDPP.get_X_y(data,use_4_features=use_4_features,normalize=normalize)
    if pca_filter > 0:
        X,a = HCPDPP.pca_filter(X,pca_filter)
        return X,y,a
    else:
        return data,X,y

def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True
    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

#
# import HCP_DataProcessor.Data_Process as HCPDPP
# import Prediction_Utils as PU
# from sklearn.model_selection import train_test_split
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# clf = LinearDiscriminantAnalysis()
# clf.fit(X_train,y_train)
# coef = clf.coef_
