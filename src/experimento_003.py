from src.experimento import normalized, sha, dump, SEED, LABELS
from src.features_003 import TextFeatures
from pathlib import Path
import json,warnings,time
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import f1_score, classification_report,confusion_matrix,make_scorer
from sklearn.exceptions import ConvergenceWarning

SCORER=make_scorer(f1_score,average='macro',labels=LABELS,zero_division=0)

def select(results):
    # Regla previa: dentro de 0.003 del mejor F1, preferir menor brecha F1.
    r=pd.DataFrame(results)
    r['f1_gap']=(r.mean_train_score-r.mean_test_score).clip(lower=0)
    eligible=r[r.mean_test_score>=r.mean_test_score.max()-0.003]
    return int(eligible.sort_values(['f1_gap','mean_test_score'],ascending=[True,False]).index[0])

def make_search(folds):
    pipe=Pipeline([('features',TextFeatures()),('clf',LinearSVC(C=1.,dual='auto',max_iter=10000,random_state=SEED))])
    baseline=joblib.load('models/modelo_002_regularizado.joblib')
    grid=[{'features__mode':['word_char','position','context','discourse'],'clf__C':[0.5,2.]},
          {'features':[clone(baseline['tfidf'])],'clf':[clone(baseline['clf'])]}]
    return GridSearchCV(pipe,grid,cv=folds,scoring=SCORER,return_train_score=True,refit=select,n_jobs=2,pre_dispatch=2,error_score='raise')

def fit_search(X,y,groups,n=3):
    folds=list(StratifiedGroupKFold(n,shuffle=True,random_state=SEED).split(X,y,groups))
    for a,b in folds: assert not set(groups.iloc[a]) & set(groups.iloc[b])
    search=make_search(folds)
    with joblib.parallel_backend('loky',inner_max_num_threads=1): search.fit(X,y)
    return search

def run():
    warnings.filterwarnings('error',category=ConvergenceWarning)
    before={str(p):sha(p) for folder in ['data','models','submissions'] for p in Path(folder).glob('*') if p.is_file()}
    train=pd.read_csv('data/train.csv',dtype={'id':str})
    parts=pd.read_csv('reports/particiones.csv',dtype={'id':str})
    assert train.id.equals(parts.id)
    ix=np.flatnonzero(parts.partition.eq('development'))
    test=np.flatnonzero(parts.partition.eq('local_test'))
    groups=normalized(train.text)
    assert not set(groups.iloc[ix])&set(groups.iloc[test])
    X=train.text.fillna('').iloc[ix].reset_index(drop=True)
    y=train.label.iloc[ix].reset_index(drop=True)
    g=groups.iloc[ix].reset_index(drop=True)
    plan={'metric':'f1_macro exclusivamente','seed':SEED,'search':'8 LinearSVC: word_char, position, context, discourse x C=[0.5,2]; referencia 002','selection':'máximo F1 macro CV; entre alternativas a <=0.003 preferir menor brecha F1 entrenamiento-validación','evaluation':'CV anidada 3 externa x 3 interna en desarrollo; prueba local original excluida','public_score_002':0.73888,'score_source':'captura proporcionada por usuario; nombre de métrica no visible en captura','goal':0.90,'limitations':'Desarrollo ya usado para experimentos y lectura de ejemplos. CV anidada separa selección de hiperparámetros de evaluación por fold, pero no es un conjunto externo virgen.'}
    dump('reports/plan_experimento_003.json',plan)
    print('Inicio selección completa en desarrollo',flush=True)
    search=fit_search(X,y,g)
    cv=pd.DataFrame(search.cv_results_)
    cv['f1_gap']=cv.mean_train_score-cv.mean_test_score
    cv['selected']=cv.index==search.best_index_
    cv.to_csv('reports/resultados_cv_003.csv',index=False)
    print(cv[['params','mean_train_score','mean_test_score','f1_gap','selected']].to_string(index=False),flush=True)
    # Evaluar el procedimiento completo, seleccionando de nuevo SOLO dentro de cada fold externo.
    outer=list(StratifiedGroupKFold(3,shuffle=True,random_state=137).split(X,y,g))
    oof=np.empty(len(y),dtype=object); baseline_oof=np.empty(len(y),dtype=object)
    nested=[]
    for k,(a,b) in enumerate(outer):
        assert not set(g.iloc[a]) & set(g.iloc[b])
        print(f'CV anidada: fold externo {k+1}/3',flush=True)
        inner=fit_search(X.iloc[a].reset_index(drop=True),y.iloc[a].reset_index(drop=True),g.iloc[a].reset_index(drop=True))
        pred=inner.predict(X.iloc[b]); oof[b]=pred
        train_pred=inner.predict(X.iloc[a])
        base=clone(joblib.load('models/modelo_002_regularizado.joblib')).fit(X.iloc[a],y.iloc[a])
        baseline_oof[b]=base.predict(X.iloc[b])
        row={'fold':k,'f1_macro_train':f1_score(y.iloc[a],train_pred,average='macro'),'f1_macro_validation':f1_score(y.iloc[b],pred,average='macro'),'f1_macro_baseline002':f1_score(y.iloc[b],baseline_oof[b],average='macro'),'selected_params':str(inner.best_params_)}
        nested.append(row)
        pd.DataFrame(inner.cv_results_).to_csv(f'reports/cv_003_interna_{k}.csv',index=False)
        print(row,flush=True)
    pd.DataFrame(nested).to_csv('reports/cv_003_anidada.csv',index=False)
    out=train.iloc[ix][['id','label']].copy();out['prediction']=oof;out['baseline002']=baseline_oof
    out['outer_fold']=-1
    for k,(_,b) in enumerate(outer):out.iloc[b,out.columns.get_loc('outer_fold')]=k
    out.to_csv('reports/predicciones_oof_003.csv',index=False)
    chosen=cv.iloc[search.best_index_]
    model=clone(search.best_estimator_).fit(train.text.fillna(''),train.label)
    joblib.dump(model,'models/modelo_003.joblib')
    ev=pd.read_csv('data/eval.csv',dtype={'id':str}); sample=pd.read_csv('data/sample_submission.csv',dtype={'id':str})
    assert ev.id.is_unique and sample.id.is_unique and set(ev.id)==set(sample.id)
    pred=model.predict(ev.text.fillna(''))
    submission=pd.DataFrame({'id':sample.id,'answer':sample.id.map(pd.Series(pred,index=ev.id))})
    assert len(submission)==len(ev)==len(sample) and submission.id.is_unique
    assert submission.id.equals(sample.id) and not submission.isna().any().any() and set(submission.answer)<=set(LABELS)
    submission.to_csv('submissions/submission_003.csv',index=False)
    pd.testing.assert_frame_equal(submission,pd.read_csv('submissions/submission_003.csv',dtype={'id':str}))
    assert np.array_equal(joblib.load('models/modelo_003.joblib').predict(ev.text.fillna('')),pred)
    assert all(sha(p)==v for p,v in before.items())
    record={'plan':plan,'selected_params':str(search.best_params_),'selected_cv_f1_macro':float(chosen.mean_test_score),'selected_cv_std':float(chosen.std_test_score),'selected_train_f1_macro':float(chosen.mean_train_score),'selected_f1_gap':float(chosen.f1_gap),'nested_f1_macro_oof':f1_score(y,oof,average='macro'),'nested_f1_macro_fold_mean':float(np.mean([r['f1_macro_validation'] for r in nested])),'nested_f1_macro_fold_std':float(np.std([r['f1_macro_validation'] for r in nested])),'nested_baseline002_f1_macro_oof':f1_score(y,baseline_oof,average='macro'),'per_class':classification_report(y,oof,labels=LABELS,output_dict=True,zero_division=0),'confusion':confusion_matrix(y,oof,labels=LABELS).tolist(),'labels':LABELS,'local_test_reused':False,'public_score_003':None,'original_files_unchanged':True,'versions':json.loads(Path('reports/registro_experimento_001.json').read_text())['versions'],'input_sha256':{p:v for p,v in before.items() if p.startswith('data/')},'artifacts_sha256':{p:sha(p) for p in ['models/modelo_003.joblib','submissions/submission_003.csv']},'checks':{'csv_ids_order_schema_labels':True,'reload_predictions':True,'roundtrip_csv':True,'disjoint_groups':True}}
    # No publicar accuracy: a partir de esta versión el informe usa solo F1.
    record['per_class'].pop('accuracy',None)
    dump('reports/registro_experimento_003.json',record)
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
    return record

if __name__=='__main__':run()
