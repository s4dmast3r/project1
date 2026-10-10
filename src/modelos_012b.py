"""Decisiones multiclase clásicas, sin acceso a eval ni datos externos."""
import numpy as np
import pandas as pd
import warnings
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.exceptions import ConvergenceWarning
from src.features_007b import TextFeatures007b
from src.experimento import normalized

class CheckedL1(LinearSVC):
    def fit(self,X,y,sample_weight=None):
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            super().fit(X,y,sample_weight=sample_weight)
        assert np.max(self.n_iter_)<self.max_iter
        return self

def svc(C=2):return CheckedL1(penalty='l1',dual=False,C=C,max_iter=20000,random_state=42)
def base():return Pipeline([('features',TextFeatures007b(thirds=True)),('clf',svc())])

class DiscountedSVC(ClassifierMixin,BaseEstimator):
    """Reproduce011 incluyendo su descuento en predict, por nombre de clase."""
    def __init__(self,discount=.2):self.discount=discount
    def fit(self,X,y):
        self.model_=base().fit(X,y);self.classes_=self.model_.classes_;return self
    def decision_function(self,X):
        s=self.model_.decision_function(X).copy()
        s[:,self.classes_!='neutral']+=self.discount
        return s
    def predict(self,X):return self.classes_[self.decision_function(X).argmax(axis=1)]

class PolaritySpecialist(ClassifierMixin,BaseEstimator):
    """Conserva puerta neutral011; sustituye polaridad por SVM binario."""
    def __init__(self,C=1):self.C=C
    def fit(self,X,y):
        y=np.asarray(y);self.base_=DiscountedSVC().fit(X,y);self.classes_=self.base_.classes_
        matrix=self.base_.model_['features'].transform(X)
        mask=y!='neutral';self.specialist_=svc(self.C).fit(matrix[mask],y[mask]);return self
    def predict(self,X):
        pred=self.base_.predict(X);mask=pred!='neutral'
        if np.any(mask):
            matrix=self.base_.model_['features'].transform(X)
            pred[mask]=self.specialist_.predict(matrix[mask])
        return pred

class GroupCalibrated(ClassifierMixin,BaseEstimator):
    """Sigmoides ajustadas sobre OOF interna; vectorizador dentro de cada fold."""
    def fit(self,X,y):
        X=pd.Series(list(X));y=np.asarray(y);g=normalized(X)
        splits=list(StratifiedGroupKFold(3,shuffle=True,random_state=271).split(X,y,g))
        for a,b in splits:assert not set(g.iloc[a])&set(g.iloc[b])
        self.model_=CalibratedClassifierCV(base(),method='sigmoid',cv=splits,ensemble=False,n_jobs=1).fit(X,y)
        self.classes_=self.model_.classes_;return self
    def predict(self,X):return self.model_.predict(X)
    def predict_proba(self,X):return self.model_.predict_proba(X)

NAMES=['ref011','polarity_c1','polarity_c2','ovo_l1','sigmoid']
def make_model(name):
    if name=='ref011':return DiscountedSVC()
    if name in ['polarity_c1','polarity_c2']:return PolaritySpecialist(C=1 if name.endswith('c1') else 2)
    if name=='ovo_l1':return Pipeline([('features',TextFeatures007b(thirds=True)),('clf',OneVsOneClassifier(svc(),n_jobs=1))])
    if name=='sigmoid':return GroupCalibrated()
    raise ValueError(name)
