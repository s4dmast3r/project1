"""Candidatos clásicos 006: stacking anidado y simplificaciones de 005."""
import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.exceptions import ConvergenceWarning
from src.features_004 import TextFeatures, discourse_tokens
from src.experimento import normalized
import pandas as pd

class CheckedSVC(LinearSVC):
    def fit(self,X,y,sample_weight=None):
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            super().fit(X,y,sample_weight=sample_weight)
        assert np.max(self.n_iter_)<self.max_iter
        return self

class CheckedLR(LogisticRegression):
    def fit(self,X,y,sample_weight=None):
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            super().fit(X,y,sample_weight=sample_weight)
        assert np.max(self.n_iter_)<self.max_iter
        return self

class NoEpilogueInteraction:
    def __call__(self,text):
        # Solo quitar interacciones; conservar segmentación, posiciones y canal global.
        return (t for t in discourse_tokens(text,max_ngram=1) if not t.startswith('epilogue'))

def positional(C=2,min_df=3,noep=False):
    features=(TfidfVectorizer(analyzer=NoEpilogueInteraction(),min_df=min_df,sublinear_tf=True,max_features=160000)
              if noep else TextFeatures(max_ngram=1,min_df=min_df,char_weight=0,max_features=160000))
    return Pipeline([('features',features),('clf',CheckedSVC(C=C,dual='auto',max_iter=10000,random_state=42))])

def words(clf):
    return Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=3,sublinear_tf=True,strip_accents='unicode',max_features=70000)),('clf',clf)])

class SafeStack(ClassifierMixin,BaseEstimator):
    """Cada fit crea sus propios folds: ninguna matriz OOF global se reutiliza."""
    def fit(self,X,y):
        g=normalized(pd.Series(list(X)))
        folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=271).split(X,y,g))
        for a,b in folds: assert not set(g.iloc[a])&set(g.iloc[b])
        # Percentiles aprendidos SOLO con scores OOF del entrenamiento de este fit.
        # transform no calcula ranks contra el lote de validación o eval.
        meta=Pipeline([('cdf',QuantileTransformer(n_quantiles=100,output_distribution='normal',random_state=42)),
                       ('clf',CheckedLR(C=1,max_iter=2500,random_state=42))])
        self.model_=StackingClassifier(estimators=[('svc',positional()),
            ('lr',words(CheckedLR(C=4,max_iter=2500,random_state=42))),
            ('nb',words(MultinomialNB(alpha=1.)))],
            final_estimator=meta,cv=folds,stack_method='auto',passthrough=False,n_jobs=1)
        self.model_.fit(X,y)
        self.classes_=self.model_.classes_
        self.n_meta_features_=self.model_.final_estimator_.n_features_in_
        return self
    def predict(self,X): return self.model_.predict(X)
    def predict_proba(self,X): return self.model_.predict_proba(X)
