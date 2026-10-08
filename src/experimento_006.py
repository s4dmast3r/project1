"""Versión 006: stacking y simplificación de 005; selección interna y CV anidada."""
from src.experimento import normalized, sha, dump, SEED, LABELS
from src.features_004 import TextFeatures
from pathlib import Path
import time, warnings, json
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import f1_score, make_scorer, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

from src.modelos_006 import positional,SafeStack

CONFIGS=[{'name':'ref005'},{'name':'c1_min5'},{'name':'no_epilogue_interaction'},{'name':'stack_svc_lr_nb'}]
RULE='Primero filtrar F1 >= referencia +0.002 y brecha <= referencia +0.002. Si no hay candidatos, conservar referencia. Entre los válidos a <=0.003 del mejor F1, menor brecha; desempate mayor F1.'

def select(results):
    df=pd.DataFrame(results);gap=(df.mean_train_score-df.mean_test_score).clip(lower=0)
    eligible=df[(gap<=gap.iloc[0]+0.002)&(df.mean_test_score>=df.mean_test_score.iloc[0]+0.002)]
    if eligible.empty:return 0
    eligible=eligible[eligible.mean_test_score>=eligible.mean_test_score.max()-0.003]
    eligible=eligible.assign(gap=gap.loc[eligible.index])
    return int(eligible.sort_values(['gap','mean_test_score'],ascending=[True,False]).index[0])

def search_fit(X,y,g):
    splits=list(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(X,y,g))
    for a,b in splits:assert not set(g.iloc[a])&set(g.iloc[b])
    pipe=Pipeline([('model',positional())])
    grid=[{'model':[positional()]}, {'model':[positional(C=1,min_df=5)]},
          {'model':[positional(noep=True)]}, {'model':[SafeStack()]}]
    search=GridSearchCV(pipe,grid,cv=splits,scoring=make_scorer(f1_score,labels=LABELS,average='macro',zero_division=0),refit=select,return_train_score=True,n_jobs=1,error_score='raise')
    search.fit(X,y)
    return search

def table(search):
    df=pd.DataFrame(search.cv_results_);df['name']=[c['name'] for c in CONFIGS]
    df['f1_gap']=df.mean_train_score-df.mean_test_score;df['selected']=df.index==search.best_index_
    return df

def run():
    start=time.perf_counter()
    old=json.loads(Path('reports/registro_005.json').read_text())
    for p,h in {**old['input_sha256'],**old['artifacts_sha256']}.items():assert sha(p)==h,p
    protected={str(p):sha(p) for folder in ['data','models','submissions','src'] for p in Path(folder).glob('*') if p.is_file()}
    train=pd.read_csv('data/train.csv',dtype={'id':str});parts=pd.read_csv('reports/particiones.csv',dtype={'id':str})
    assert train.id.equals(parts.id)
    dev=np.flatnonzero(parts.partition.eq('development'));test=np.flatnonzero(parts.partition.eq('local_test'))
    groups=normalized(train.text);assert not set(groups.iloc[dev])&set(groups.iloc[test])
    X=train.text.fillna('').iloc[dev].reset_index(drop=True);y=train.label.iloc[dev].reset_index(drop=True);g=groups.iloc[dev].reset_index(drop=True)
    dump('reports/plan_006.json',{'configs':CONFIGS,'rule':RULE,'inner_seed':42,'outer_seed':137,'nested_cv':'3x3, desarrollo anterior; no prueba virgen ni corrección total del sesgo histórico','historical_local_test':'excluida de selección y evaluación','n_jobs':1,'resource_choice':'Un proceso para limitar RAM; un hilo numérico configurado antes de importar numpy mediante src.experimento','goal':'mejor F1 macro sin aumentar materialmente brecha frente a 005; stacking con CDF aprendida solo sobre OOF de cada entrenamiento'})
    print('Selección en desarrollo: cuatro configuraciones',flush=True)
    search=search_fit(X,y,g);cv=table(search);cv.to_csv('reports/resultados_cv_006.csv',index=False)
    print(cv[['name','mean_train_score','mean_test_score','f1_gap','selected']].to_string(index=False),flush=True)
    outer=list(StratifiedGroupKFold(3,shuffle=True,random_state=137).split(X,y,g))
    oof=np.empty(len(y),object);ref_oof=np.empty(len(y),object);fold_ids=np.full(len(y),-1);rows=[]
    for k,(a,b) in enumerate(outer):
        assert not set(g.iloc[a]) & set(g.iloc[b])
        print(f'Fold externo {k+1}/3: selección solamente dentro del entrenamiento',flush=True)
        inner=search_fit(X.iloc[a].reset_index(drop=True),y.iloc[a].reset_index(drop=True),g.iloc[a].reset_index(drop=True))
        table(inner).to_csv(f'reports/cv_006_interna_{k}.csv',index=False)
        oof[b]=inner.predict(X.iloc[b]);fold_ids[b]=k
        reference=clone(joblib.load('models/modelo_005.joblib')).fit(X.iloc[a],y.iloc[a])
        assert np.max(reference['clf'].n_iter_)<reference['clf'].max_iter
        ref_oof[b]=reference.predict(X.iloc[b])
        row={'fold':k,'selected':CONFIGS[inner.best_index_]['name'],'f1_train':f1_score(y.iloc[a],inner.predict(X.iloc[a]),average='macro',labels=LABELS),'f1_external':f1_score(y.iloc[b],oof[b],average='macro',labels=LABELS),'reference005_external':f1_score(y.iloc[b],ref_oof[b],average='macro',labels=LABELS)}
        rows.append(row);print(row,flush=True)
    folds=pd.DataFrame(rows);folds.to_csv('reports/cv_006_anidada.csv',index=False)
    pd.DataFrame({'id':train.id.iloc[dev].to_numpy(),'label':y,'outer_fold':fold_ids,'prediction':oof,'reference005':ref_oof}).to_csv('reports/predicciones_oof_006.csv',index=False)
    perclass=classification_report(y,oof,labels=LABELS,output_dict=True,zero_division=0);perclass.pop('accuracy',None)
    chosen=cv.iloc[search.best_index_]
    rec={'rule':RULE,'selected_config':CONFIGS[search.best_index_],'selection_f1':float(chosen.mean_test_score),'selection_std':float(chosen.std_test_score),'train_f1':float(chosen.mean_train_score),'gap':float(chosen.f1_gap),'baseline_selection_f1':float(cv.iloc[0].mean_test_score),'baseline_gap':float(cv.iloc[0].f1_gap),'nested_oof_f1':f1_score(y,oof,average='macro',labels=LABELS),'baseline_nested_oof_f1':f1_score(y,ref_oof,average='macro',labels=LABELS),'external_fold_mean':float(folds.f1_external.mean()),'external_fold_std':float(folds.f1_external.std(ddof=0)),'per_class':perclass,'confusion':confusion_matrix(y,oof,labels=LABELS).tolist(),'labels':LABELS,'input_sha256':old['input_sha256'],'versions':old['versions'],'local_test_reused':False,'public_score_006':None,'limitations':'Desarrollo y folds ya examinados. La comparación de 006 separa folds de selección y entrenamiento de meta-modelos; no borra la adaptación histórica al desarrollo. No hay prueba externa nueva.'}
    # Solo producir nueva versión si la selección interna eligió un cambio real.
    if search.best_index_==0:
        rec['submission_created']=False;rec['reason']='No hay candidato que cumpla la regla previa de mejora y brecha.'
    else:
        model=clone(search.best_estimator_).fit(train.text.fillna(''),train.label)
        joblib.dump(model,'models/modelo_006.joblib')
        ev=pd.read_csv('data/eval.csv',dtype={'id':str});sample=pd.read_csv('data/sample_submission.csv',dtype={'id':str})
        assert ev.id.is_unique and sample.id.is_unique and set(ev.id)==set(sample.id)
        predictions=model.predict(ev.text.fillna(''))
        sub=pd.DataFrame({'id':sample.id,'answer':sample.id.map(pd.Series(predictions,index=ev.id))})
        assert len(sub)==len(ev)==len(sample) and sub.id.is_unique and sub.id.equals(sample.id)
        assert not sub.isna().any().any() and set(sub.answer)<=set(LABELS)
        sub.to_csv('submissions/submission_006.csv',index=False)
        pd.testing.assert_frame_equal(sub,pd.read_csv('submissions/submission_006.csv',dtype={'id':str}))
        assert np.array_equal(joblib.load('models/modelo_006.joblib').predict(ev.text.fillna('')),predictions)
        rec.update(submission_created=True,artifacts_sha256={p:sha(p) for p in ['models/modelo_006.joblib','submissions/submission_006.csv']},checks={'ids_order_schema_labels':True,'roundtrip':True,'reload_predictions':True,'disjoint_groups':True})
    assert all(sha(p)==h for p,h in protected.items())
    rec['old_artifacts_unchanged']=True;rec['elapsed_seconds']=time.perf_counter()-start
    rec['source_sha256']={p:sha(p) for p in ['src/experimento_006.py','src/modelos_006.py','src/features_004.py','reports/particiones.csv']}
    dump('reports/registro_006.json',rec)
    print(json.dumps(rec,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':run()
