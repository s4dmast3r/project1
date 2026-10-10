"""NLP clásico 007: canales de negación/contraste sin reglas de etiqueta."""
import re
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from src.features_004 import TextFeatures,clean

class ScopeAnalyzer:
    def __init__(self,mode='neg_scope'):self.mode=mode
    def __call__(self,text):
        text=clean(text)
        if self.mode=='neg_scope':
            remaining=0;previous=None
            for token in re.findall(r'\w+|[^\w\s]',text):
                if token in {'no','nunca','sin','ni','jamas','tampoco'}:
                    remaining=3;previous=None
                    continue
                if not token.isalnum() or token in {'pero','aunque','sin','embargo'}:
                    remaining=0;previous=None
                    continue
                if remaining:
                    yield 'neg_'+token
                    if previous is not None:yield 'neg_'+previous+'_'+token
                    previous=token;remaining-=1
        elif self.mode=='post_contraste':
            for sentence in re.split(r'[.!?;]+',text):
                pieces=re.split(r'\b(?:pero|aunque|sin embargo)\b',sentence)
                for clause in pieces[1:]:
                    tokens=re.findall(r'\b\w+\b',clause)
                    for i,t in enumerate(tokens):
                        yield 'post_'+t
                        if i:yield 'post_'+tokens[i-1]+'_'+t
        else:raise ValueError(self.mode)

class ExtraFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,mode='neg_scope',weight=0.35):
        self.mode=mode;self.weight=weight
    def fit(self,X,y=None):
        self.union_=FeatureUnion([
            ('base',TextFeatures(max_ngram=1,min_df=3,char_weight=0,max_features=160000)),
            ('extra',TfidfVectorizer(analyzer=ScopeAnalyzer(self.mode),min_df=3,sublinear_tf=True,max_features=30000))],
            transformer_weights={'base':1.,'extra':self.weight})
        self.union_.fit(X)
        return self
    def transform(self,X):return self.union_.transform(X)
