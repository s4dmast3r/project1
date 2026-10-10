"""Ronda 007: CV agrupada repetida; todos los scores sirven para selección."""
from src.experimento import normalized,sha,dump,LABELS
from src.features_004 import TextFeatures
from src.features_007 import ExtraFeatures
from pathlib import Path
import time,warnings,json
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score,classification_report,confusion_matrix
from sklearn.exceptions import ConvergenceWarning

SEEDS=[271,599,913]
NAMES=['ref005','neg_scope','post_contraste','c1_min5']

def make_model(name):
    if name in ['neg_scope','post_contraste']:
        features=ExtraFeatures(mode=name,weight=0.35);C=2
    else:
        features=TextFeatures(max_ngram=1,min_df=5 if name=='c1_min5' else 3,char_weight=0,max_features=160000)
        C=1 if name=='c1_min5' else 2
    return Pipeline([('features',features),('clf',LinearSVC(C=C,dual='auto',max_iter=10000,random_state=42))])

def fit(pipe,X,y):
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        pipe.fit(X,y)
    assert np.max(pipe['clf'].n_iter_)<pipe['clf'].max_iter
    return pipe

def score(y,p):return float(f1_score(y,p,average='macro',labels=LABELS,zero_division=0))

def gates(oof_delta,fold_deltas,gap_delta):
    wins=int(np.count_nonzero(np.asarray(fold_deltas)>0))
    return bool(oof_delta>=0.005 and (wins==3 or (wins>=2 and gap_delta<0)))

def run():
    started=time.perf_counter()
    if Path('reports/registro_007.json').exists():raise FileExistsError('Conservar resultados 007: archivar explícitamente antes de repetir.')
    old=json.loads(Path('reports/registro_005.json').read_text())
    for p,h in {**old['input_sha256'],**old['artifacts_sha256'],**old['source_sha256']}.items():assert sha(p)==h,p
    protected={str(p):sha(p) for folder in ['data','models','submissions','src'] for p in Path(folder).glob('*') if p.is_file()}
    train=pd.read_csv('data/train.csv',dtype={'id':str});parts=pd.read_csv('reports/particiones.csv',dtype={'id':str})
    assert train.id.equals(parts.id)
    d=parts.partition.eq('development');gall=normalized(train.text)
    assert not set(gall[d])&set(gall[~d])
    X=train.text.fillna('')[d].reset_index(drop=True);y=train.label[d].reset_index(drop=True);g=gall[d].reset_index(drop=True)
    ids=train.id[d].reset_index(drop=True)
    plan={'seeds':SEEDS,'folds_per_seed':3,'candidates':NAMES,'metric':'f1_macro; labels explícitos; zero_division=0','selection':'Cada semilla: delta F1 OOF >=0.005 vs005 y ganar 3 folds o >=2 con menor brecha media. Debe cumplir las tres semillas. Luego no perder F1 OOF en semilla histórica137. Entre aprobados mayor F1 medio en semillas nuevas; empate exacto menor brecha.','historical_validation':'137 se usa como puerta de selección adicional, no prueba independiente.','data_scope':'9600 filas de desarrollo histórico. Prueba local antigua excluida de selección.','extra_features':{'scope_tokens':3,'negators':['no','nunca','sin','ni','jamas','tampoco'],'contrast':['pero','aunque','sin embargo'],'extra_weight':0.35,'extra_min_df':3,'extra_max_features':30000,'ngram_range':[1,2]},'limitations':'Cambiar semillas no crea datos vírgenes. 271 ya se usó dentro de stacking006. Todos estos scores se usan para selección, no son evaluación externa independiente.','budget':'36 ajustes de screening; hasta12 históricos condicionales; un ajuste final solo si aprobado','n_jobs':1}
    dump('reports/plan_007.json',plan)
    rows=[];summaries=[];predictions=[];all_oof={}
    def evaluate(name,seed):
        pred=np.empty(len(y),dtype=object);fold_index=np.full(len(y),-1);local_rows=[]
        for k,(a,b) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(X,y,g)):
            assert not set(g.iloc[a])&set(g.iloc[b])
            t=time.perf_counter();model=fit(make_model(name),X.iloc[a],y.iloc[a])
            pred[b]=model.predict(X.iloc[b]);fold_index[b]=k
            tr=score(y.iloc[a],model.predict(X.iloc[a]));va=score(y.iloc[b],pred[b])
            row={'name':name,'seed':seed,'fold':k,'train_f1':tr,'validation_f1':va,'gap':tr-va,'iterations':int(model['clf'].n_iter_),'seconds':time.perf_counter()-t}
            rows.append(row);local_rows.append(row)
        s={'name':name,'seed':seed,'oof_f1':score(y,pred),'fold_mean':float(np.mean([r['validation_f1'] for r in local_rows])),'fold_std':float(np.std([r['validation_f1'] for r in local_rows])),'train_mean':float(np.mean([r['train_f1'] for r in local_rows])),'gap':float(np.mean([r['gap'] for r in local_rows]))}
        summaries.append(s);all_oof[name,seed]=pred
        predictions.append(pd.DataFrame({'id':ids,'label':y,'name':name,'seed':seed,'fold':fold_index,'prediction':pred}))
        pd.DataFrame(rows).to_csv('reports/folds_007.csv',index=False)
        pd.DataFrame(summaries).to_csv('reports/resultados_cv_007.csv',index=False)
        print(s,flush=True)
        return s
    for seed in SEEDS:
        for name in NAMES:evaluate(name,seed)
    results=pd.DataFrame(summaries);fold_results=pd.DataFrame(rows);checks=[];eligible=[]
    for name in NAMES[1:]:
        passed=True
        for seed in SEEDS:
            ref=results[(results.name=='ref005')&(results.seed==seed)].iloc[0]
            cand=results[(results.name==name)&(results.seed==seed)].iloc[0]
            ref_f=fold_results[(fold_results.name=='ref005')&(fold_results.seed==seed)].sort_values('fold')
            c_f=fold_results[(fold_results.name==name)&(fold_results.seed==seed)].sort_values('fold')
            deltas=c_f.validation_f1.to_numpy()-ref_f.validation_f1.to_numpy()
            delta=float(cand.oof_f1-ref.oof_f1);gd=float(cand.gap-ref.gap)
            ok=gates(delta,deltas,gd);passed=passed and ok
            checks.append({'name':name,'seed':seed,'f1_gain':delta,'folds_won':int((deltas>0).sum()),'gap_change':gd,'passes':ok})
        if passed:eligible.append(name)
    pd.DataFrame(checks).to_csv('reports/criterios_007.csv',index=False)
    historical={};accepted=[]
    if eligible:
        ref=evaluate('ref005',137)
        for name in eligible:
            cand=evaluate(name,137);historical[name]={'candidate':cand['oof_f1'],'reference':ref['oof_f1'],'passes':cand['oof_f1']>=ref['oof_f1']}
            if historical[name]['passes']:accepted.append(name)
    pd.concat(predictions,ignore_index=True).to_csv('reports/predicciones_oof_007.csv',index=False)
    winner=None
    if accepted:
        ordering=results[results.name.isin(accepted)].groupby('name').agg(f1=('oof_f1','mean'),gap=('gap','mean')).sort_values(['f1','gap'],ascending=[False,True])
        winner=str(ordering.index[0])
    rec={'plan':plan,'winner':winner,'submission_created':False,'criteria':checks,'historical':historical,'seed_results':summaries,'reference_public005':0.88111,'public006':'aproximadamente0.87 según documento del auditor; no se aportó valor exacto/captura en este turno','public007':None,'input_sha256':old['input_sha256'],'versions':old['versions'],'local_test_reused':False,'evaluation_type':'CV repetida usada para selección; no evaluación externa independiente'}
    if winner:
        model=fit(make_model(winner),train.text.fillna(''),train.label)
        joblib.dump(model,'models/modelo_007.joblib')
        ev=pd.read_csv('data/eval.csv',dtype={'id':str});sample=pd.read_csv('data/sample_submission.csv',dtype={'id':str})
        assert ev.id.is_unique and sample.id.is_unique and set(ev.id)==set(sample.id)
        p=model.predict(ev.text.fillna(''))
        sub=pd.DataFrame({'id':sample.id,'answer':sample.id.map(pd.Series(p,index=ev.id))})
        assert len(sub)==len(ev)==len(sample) and sub.id.is_unique and sub.id.equals(sample.id)
        assert not sub.isna().any().any() and set(sub.answer)<=set(LABELS)
        sub.to_csv('submissions/submission_007.csv',index=False)
        pd.testing.assert_frame_equal(sub,pd.read_csv('submissions/submission_007.csv',dtype={'id':str}))
        assert np.array_equal(joblib.load('models/modelo_007.joblib').predict(ev.text.fillna('')),p)
        rec.update(submission_created=True,artifacts_sha256={p:sha(p) for p in ['models/modelo_007.joblib','submissions/submission_007.csv']},csv_checks_passed=True)
    else:
        rec['decision']='Conservar005: ninguna candidata cumple todos los criterios previos. No generar un CSV equivalente ni reducir los umbrales después de observar resultados.'
    assert all(sha(p)==v for p,v in protected.items())
    rec['old_artifacts_unchanged']=True;rec['elapsed_seconds']=time.perf_counter()-started
    rec['source_sha256']={p:sha(p) for p in ['src/experimento_007.py','src/features_007.py','src/features_004.py','reports/particiones.csv']}
    dump('reports/registro_007.json',rec)
    print(json.dumps({'winner':winner,'submission_created':rec['submission_created'],'decision':rec.get('decision'),'seconds':rec['elapsed_seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':run()
