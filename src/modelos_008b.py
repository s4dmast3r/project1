"""Métodos clásicos 008b; ver referencias y adaptaciones en investigacion_008b.md."""
import re,warnings
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin,ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from src.features_007b import DiscourseAnalyzer,discourse_tokens,clean,TextFeatures007b

class WeightedThirds(BaseEstimator,TransformerMixin):
    def __init__(self,weight=1.):self.weight=weight
    def fit(self,X,y=None):
        self.vectorizer_=TfidfVectorizer(analyzer=DiscourseAnalyzer(thirds=True),min_df=3,max_features=160000,sublinear_tf=True).fit(X)
        self.weights_=np.array([self.weight if t.startswith('third') else 1. for t in self.vectorizer_.get_feature_names_out()])
        return self
    def transform(self,X):return normalize(self.vectorizer_.transform(X).multiply(self.weights_).tocsr(),copy=False)

class ConnectorAnalyzer:
    def __call__(self,text):
        yield from discourse_tokens(text,thirds=True)
        # Características palabra×conector; nunca asigna una polaridad al conector.
        for sentence in re.split(r'[.!?;]+',clean(text)):
            parts=re.split(r'\b(pero|aunque|sin embargo|eso si|aun asi|por cierto|de paso|por otro lado)\b',sentence)
            for k in range(1,len(parts)-1,2):
                for word in re.findall(r'\b\w+\b',parts[k+1]):yield 'connector_'+parts[k].replace(' ','_')+':'+word

class NBRatioSVM(ClassifierMixin,BaseEstimator):
    """Adaptación multiclase uno-contra-resto del NB-SVM, sin interpolación."""
    def __init__(self,C=1.,alpha=1.):self.C=C;self.alpha=alpha
    def fit(self,X,y):
        X=X.tocsr().copy();X.data[:]=1.;y=np.asarray(y)
        self.classes_=np.unique(y);self.ratios_=[];self.models_=[]
        for label in self.classes_:
            target=y==label
            p=np.asarray(X[target].sum(axis=0)).ravel()+self.alpha
            q=np.asarray(X[~target].sum(axis=0)).ravel()+self.alpha
            ratio=np.log(p/p.sum())-np.log(q/q.sum())
            model=LinearSVC(C=self.C,dual='auto',max_iter=20000,random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                model.fit(X.multiply(ratio).tocsr(),target)
            assert model.n_iter_<model.max_iter
            self.ratios_.append(ratio);self.models_.append(model)
        self.n_iter_=np.array([m.n_iter_ for m in self.models_]);return self
    def decision_function(self,X):
        X=X.tocsr().copy();X.data[:]=1.
        return np.column_stack([m.decision_function(X.multiply(r).tocsr()) for m,r in zip(self.models_,self.ratios_)])
    def predict(self,X):return self.classes_[np.argmax(self.decision_function(X),axis=1)]
