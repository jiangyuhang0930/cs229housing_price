from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

def pca_selection(X, y, k):
    selector = PCA(n_components=k)
    X_reduced = selector.fit_transform(X)
    return X_reduced, selector

def f_selection(X, y, k):
    selector = SelectKBest(f_classif, k=k)
    X_reduced = selector.fit_transform(X, y)
    # print(list(X.iloc[:, selector.get_support(indices=True)].columns.values))
    return X_reduced, selector

