"""Ronda acotada frente a010; accuracy primaria autorizada por usuario."""
from src.experimento import normalized,sha,dump,LABELS
from src.modelos_011b import make_model,NAMES
from pathlib import Path
import json,time,warnings,sys
import numpy as np
import pandas as pd
import joblib,sklearn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from sklearn.exceptions import ConvergenceWarning
SEEDS=[8803,9901,11003]

def metrics(y,p):
    return {'accuracy':float(accuracy_score(y,p)),'f1':float(f1_score(y,p,average='macro',labels=LABELS,zero_division=0))}

def fit(model,x,y):
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(x,y)
    clf=model['clf']; estimators=clf.estimators_ if hasattr(clf,'estimators_') else [clf]
    for est in estimators:assert np.max(est.n_iter_)<est.max_iter
    return model

def run():
    started=time.perf_counter()
    for p in ['reports/registro_011b.json','models/modelo_011.joblib','submissions/submission_011.csv']:
        if Path(p).exists():raise FileExistsError(p)
    old=json.loads(Path('reports/registro_013.json').read_text())
    inputs=json.loads(Path('reports/registro_005.json').read_text())['input_sha256']
    for p,h in {**inputs,**old['source_sha256'],**old['artifacts_sha256']}.items():assert sha(p)==h,p
    protected={str(p):sha(p) for folder in ['data','models','submissions','src'] for p in Path(folder).glob('*') if p.is_file()}
    train=pd.read_csv('data/train.csv',dtype={'id':str}); parts=pd.read_csv('reports/particiones.csv',dtype={'id':str})
    assert train.id.equals(parts.id) and train.id.is_unique
    d=parts.partition.eq('development');groups=normalized(train.text)
    assert not set(groups[d])&set(groups[~d])
    X=train.text[d].reset_index(drop=True);y=train.label[d].reset_index(drop=True);g=groups[d].reset_index(drop=True);ids=train.id[d].reset_index(drop=True)
    plan={'date':'2026-10-09','primary_metric':'accuracy','secondary':'F1 macro','authorization':'Usuario confirmó accuracy principal y F1 macro control en este chat.',
          'candidates':NAMES,'seeds':SEEDS,'folds':3,'historical_seed':137,
          'gates':'>=7/9 folds accuracy; delta medio accuracy >=0.003; delta F1 medio >=0; brecha accuracy media no mayor; no perder accuracy ni F1 en137.',
          'selection':'Mayor accuracy media entre elegibles; empate menor brecha. Sin rondas nuevas ni cambios de criterio después de observar.',
          'budget':'54 ajustes de pipeline; hasta18 adicionales en137 si hay elegibles; un ajuste final si aprueba.',
          'limitations':'Se reutilizan particiones de013 para comparación. Todos los scores son de selección sobre datos ya explorados; no prueba independiente.',
          'sources':'reports/investigacion_011b.md'}
    dump('reports/plan_011b.json',plan)
    rows=[];summaries=[];predictions=[]
    def evaluate(name,seed):
        pred=np.empty(len(y),dtype=object);foldid=np.full(len(y),-1);local=[]
        for k,(a,b) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(X,y,g)):
            assert not set(g.iloc[a])&set(g.iloc[b]);t=time.perf_counter()
            model=fit(make_model(name),X.iloc[a],y.iloc[a]);pred[b]=model.predict(X.iloc[b]);foldid[b]=k
            tr=metrics(y.iloc[a],model.predict(X.iloc[a]));va=metrics(y.iloc[b],pred[b])
            row={'name':name,'seed':seed,'fold':k,'train_accuracy':tr['accuracy'],'validation_accuracy':va['accuracy'],
                 'train_f1':tr['f1'],'validation_f1':va['f1'],'gap':tr['accuracy']-va['accuracy'],'seconds':time.perf_counter()-t}
            rows.append(row);local.append(row)
        s={'name':name,'seed':seed,**metrics(y,pred),'gap':float(np.mean([r['gap'] for r in local]))}
        summaries.append(s);predictions.append(pd.DataFrame({'id':ids,'label':y,'name':name,'seed':seed,'fold':foldid,'prediction':pred}))
        pd.DataFrame(rows).to_csv('reports/folds_011b.csv',index=False)
        pd.DataFrame(summaries).to_csv('reports/resultados_011b.csv',index=False)
        print(s,flush=True);return s
    for seed in SEEDS:
        for name in NAMES:evaluate(name,seed)
    results=pd.DataFrame(summaries);folds=pd.DataFrame(rows);checks=[];eligible=[]
    ref=results[results.name=='ref010'].set_index('seed')
    rf=folds[folds.name=='ref010'].set_index(['seed','fold'])
    for name in NAMES[1:]:
        cand=results[results.name==name].set_index('seed');cf=folds[folds.name==name].set_index(['seed','fold'])
        gain=float((cand.accuracy-ref.accuracy).mean());fg=float((cand.f1-ref.f1).mean());gap=float((cand.gap-ref.gap).mean())
        wins=int((cf.validation_accuracy>rf.validation_accuracy).sum())
        passed=bool(wins>=7 and gain>=.003 and fg>=0 and gap<=0)
        checks.append({'name':name,'accuracy_gain':gain,'f1_gain':fg,'gap_change':gap,'folds_won':wins,'passes':passed})
        if passed:eligible.append(name)
    pd.DataFrame(checks).to_csv('reports/criterios_011b.csv',index=False)
    historical={};accepted=[]
    if eligible:
        r=evaluate('ref010',137)
        for name in eligible:
            c=evaluate(name,137);passed=c['accuracy']>=r['accuracy'] and c['f1']>=r['f1']
            historical[name]={'reference':r,'candidate':c,'passes':passed}
            if passed:accepted.append(name)
    pd.concat(predictions,ignore_index=True).to_csv('reports/predicciones_oof_011b.csv',index=False)
    winner=None
    if accepted:
        winner=str(results[results.name.isin(accepted)].groupby('name').agg(accuracy=('accuracy','mean'),gap=('gap','mean')).sort_values(['accuracy','gap'],ascending=[False,True]).index[0])
    rec={'plan':plan,'winner':winner,'submission_created':False,'criteria':checks,'historical':historical,'seed_results':summaries,
         'input_sha256':inputs,'versions':{'python':sys.version,'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__,'joblib':joblib.__version__},
         'public010':0.89777,'public011':None,'local_test_used_for_selection':False}
    if winner:
        model=fit(make_model(winner),train.text,train.label)
        ev=pd.read_csv('data/eval.csv',dtype={'id':str});sample=pd.read_csv('data/sample_submission.csv',dtype={'id':str})
        assert ev.id.is_unique and sample.id.is_unique and set(ev.id)==set(sample.id)
        predictions_eval=model.predict(ev.text)
        sub=pd.DataFrame({'id':sample.id,'answer':sample.id.map(pd.Series(predictions_eval,index=ev.id))})
        assert len(sub)==len(ev)==3000 and sub.id.equals(sample.id) and not sub.isna().any().any() and set(sub.answer)<=set(LABELS)
        joblib.dump(model,'models/modelo_011.joblib');sub.to_csv('submissions/submission_011.csv',index=False)
        assert np.array_equal(joblib.load('models/modelo_011.joblib').predict(ev.text),predictions_eval)
        pd.testing.assert_frame_equal(sub,pd.read_csv('submissions/submission_011.csv',dtype={'id':str}))
        rec.update(submission_created=True,artifacts_sha256={p:sha(p) for p in ['models/modelo_011.joblib','submissions/submission_011.csv']},csv_checks_passed=True)
    else:rec['decision']='Conservar010: ninguna candidata supera los criterios. No crear un envío equivalente ni bajar umbrales a posteriori.'
    assert all(sha(p)==h for p,h in protected.items())
    rec['old_artifacts_unchanged']=True;rec['elapsed_seconds']=time.perf_counter()-started
    rec['source_sha256']={p:sha(p) for p in ['src/experimento_011b.py','src/modelos_011b.py','src/modelos_008b.py','src/features_007b.py','reports/particiones.csv']}
    dump('reports/registro_011b.json',rec)
    print(json.dumps({'winner':winner,'submission_created':rec['submission_created'],'seconds':rec['elapsed_seconds']},ensure_ascii=False),flush=True)
if __name__=='__main__':run()
