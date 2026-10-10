"""Alternativas clásicas a 010; ningún recurso externo de entrenamiento."""
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from src.features_007b import TextFeatures007b,DiscourseAnalyzer
from src.modelos_008b import ConnectorAnalyzer

NAMES=['ref010','binary_idf','binary_noidf','tf_noidf','connector_l1','logistic_l1']
def make_model(name):
    clf=LinearSVC(penalty='l1',dual=False,C=2,max_iter=20000,random_state=42)
    feats=TextFeatures007b(thirds=True)
    if name in ['binary_idf','binary_noidf','tf_noidf']:
        feats=TfidfVectorizer(analyzer=DiscourseAnalyzer(thirds=True),min_df=3,max_features=160000,
             binary=name.startswith('binary'),use_idf=name=='binary_idf',sublinear_tf=name=='tf_noidf')
    elif name=='connector_l1':
        feats=TfidfVectorizer(analyzer=ConnectorAnalyzer(),min_df=3,max_features=160000,sublinear_tf=True)
    elif name=='logistic_l1':
        clf=OneVsRestClassifier(LogisticRegression(solver='liblinear',l1_ratio=1,C=4,max_iter=3000,random_state=42),n_jobs=1)
    elif name!='ref010':raise ValueError(name)
    return Pipeline([('features',feats),('clf',clf)])
