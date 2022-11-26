from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import ExtraTreesRegressor
import my_models

def pca_selection(X, y, k):
    selector = PCA(n_components=k)
    X_reduced = selector.fit_transform(X)
    return X_reduced, selector

def f_selection(X, y, k):
    selector = SelectKBest(f_regression, k=k)
    X_reduced = selector.fit_transform(X, y)
    print(list(X.iloc[:, selector.get_support(indices=True)].columns.values))
    return X_reduced, selector

def lasso_rfe_selection(X, y, k):
    model = my_models.get_lasso_model()
    selector = RFE(model, n_features_to_select=k, step=1)
    X_reduced = selector.fit_transform(X, y)
    print(list(X.iloc[:, selector.get_support(indices=True)].columns.values))
    return X_reduced, selector

def extra_tree_rfe_selection(X, y, k):
    model = ExtraTreesRegressor()
    selector = RFE(model, n_features_to_select=k, step=5)
    X_reduced = selector.fit_transform(X, y)
    print(list(X.iloc[:, selector.get_support(indices=True)].columns.values))
    return X_reduced, selector

