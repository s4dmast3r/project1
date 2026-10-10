"""Ronda 015: aislamiento del efecto de selección vs interacciones (diagnóstico).

Transformadores reutilizables:
- L1Selector: recorta a las k características con mayor |coeficiente| de un SVM L1
  ajustado dentro del mismo entrenamiento.
- PolyBlock: interacciones de grado 2 (interaction_only) sobre las k seleccionadas,
  con el bloque renormalizado L2 por fila para controlar la escala (los productos
  TF-IDF pequeños quedarían despreciados por L1 de otro modo).
- FullPlusPoly: representación completa + el bloque de interacciones normalizado.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, normalize
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
import warnings


class L1Selector(BaseEstimator, TransformerMixin):
    def __init__(self, k=500, C=2.0):
        self.k = k
        self.C = C

    def fit(self, X, y=None):
        m = LinearSVC(penalty='l1', dual=False, C=self.C, max_iter=20000, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            m.fit(X, y)
        assert np.max(m.n_iter_) < m.max_iter
        imp = np.abs(m.coef_).max(axis=0)
        self.idx_ = np.argsort(-imp)[:self.k]
        return self

    def transform(self, X):
        return X[:, self.idx_]


class PolyBlock(BaseEstimator, TransformerMixin):
    """Polinomio grado 2 (interaction_only) sobre top-k, bloque L2-normalizado por fila."""

    def __init__(self, k=500, C=2.0):
        self.k = k
        self.C = C

    def fit(self, X, y=None):
        self.sel_ = L1Selector(k=self.k, C=self.C).fit(X, y)
        self.poly_ = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.poly_.fit(self.sel_.transform(X))
        return self

    def transform(self, X):
        P = self.poly_.transform(self.sel_.transform(X))
        return normalize(P, norm='l2', copy=False)


class FullPlusPoly(BaseEstimator, TransformerMixin):
    def __init__(self, k=500, C=2.0):
        self.k = k
        self.C = C

    def fit(self, X, y=None):
        self.block_ = PolyBlock(k=self.k, C=self.C).fit(X, y)
        return self

    def transform(self, X):
        return sp.hstack([X, self.block_.transform(X)], format='csr')
