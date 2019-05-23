import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def pca_filter(X,n):
    pca = PCA(n_components=n)
    pca.fit(X)
    pd_pca = pd.DataFrame(pca.transform(X))
    return pd_pca, pca.explained_variance_
