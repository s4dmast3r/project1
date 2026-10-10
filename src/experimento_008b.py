"""Ronda 008b: CV agrupada repetida; todos los scores sirven para selección."""
from src.experimento import normalized,sha,dump,LABELS
from src.features_004 import TextFeatures
from src.modelos_008b import WeightedThirds,ConnectorAnalyzer,NBRatioSVM
from src.features_007b import TextFeatures007b,DiscourseAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_selection import SelectPercentile,chi2
from sklearn.preprocessing import Normalizer
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

SEEDS=[711,1237,2003]
NAMES=['ref007','joint_w05','joint_w15','connector_identity','chi2_50','nbsvm']

def make_model(name):
    clf=LinearSVC(C=2,dual='auto',max_iter=10000,random_state=42)
    base=TextFeatures007b(thirds=True)
    if name=='ref007':return Pipeline([('features',base),('clf',clf)])
    if name in ['joint_w05','joint_w15']:
        return Pipeline([('features',WeightedThirds(.5 if name=='joint_w05' else 1.5)),('clf',clf)])
    if name=='connector_identity':return Pipeline([('features',TfidfVectorizer(analyzer=ConnectorAnalyzer(),min_df=3,max_features=160000,sublinear_tf=True)),('clf',clf)])
    if name=='chi2_50':return Pipeline([('features',base),('select',SelectPercentile(chi2,percentile=50)),('norm',Normalizer()),('clf',clf)])
    if name=='nbsvm':return Pipeline([('features',CountVectorizer(analyzer=DiscourseAnalyzer(thirds=True),binary=True,min_df=3,max_features=160000)),('clf',NBRatioSVM())])
    raise ValueError(name)

def fit(pipe,X,y):
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        pipe.fit(X,y)
    assert np.max(pipe['clf'].n_iter_)<getattr(pipe['clf'],'max_iter',20000)
    return pipe

def score(y,p):return float(f1_score(y,p,average='macro',labels=LABELS,zero_division=0))

def run():
    started=time.perf_counter()
    if Path('reports/registro_008b.json').exists():raise FileExistsError('Conservar resultados 007: archivar explícitamente antes de repetir.')
    old=json.loads(Path('reports/registro_confirmacion_007b.json').read_text())
    inputs=json.loads(Path('reports/registro_005.json').read_text())['input_sha256']
    for p,h in {**inputs,**old['artifacts_sha256'],**old['source_sha256']}.items():assert sha(p)==h,p
    protected={str(p):sha(p) for folder in ['data','models','submissions','src'] for p in Path(folder).glob('*') if p.is_file()}
    train=pd.read_csv('data/train.csv',dtype={'id':str});parts=pd.read_csv('reports/particiones.csv',dtype={'id':str})
    assert train.id.equals(parts.id)
    d=parts.partition.eq('development');gall=normalized(train.text)
    assert not set(gall[d])&set(gall[~d])
    X=train.text.fillna('')[d].reset_index(drop=True);y=train.label[d].reset_index(drop=True);g=gall[d].reset_index(drop=True)
    ids=train.id[d].reset_index(drop=True)
    plan={'seeds':SEEDS,'folds_per_seed':3,'candidates':NAMES,'metric':'F1 macro exclusivamente',
        'gates':'>=7/9 folds ganados; delta medio OOF >=0.003; brecha media no mayor; no perder en137. No alterar después de ver resultados.',
        'budget':'54 ajustes de screening y hasta18 históricos condicionales; sin nuevas rondas adaptativas',
        'references':'reports/investigacion_008b.md',
        'limitations':'CV repetida usada para selección. Semillas/datos ya explorados; ninguna prueba nueva.',
        'hypotheses':'Ponderar tercios en el MISMO TFIDF; chi2 supervisado dentro del fold; conector identificado; NB-SVM multiclase adaptado.'}
    dump('reports/plan_008b.json',plan)
    rows=[];summaries=[];predictions=[];all_oof={}
    def evaluate(name,seed):
        pred=np.empty(len(y),dtype=object);fold_index=np.full(len(y),-1);local_rows=[]
        for k,(a,b) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(X,y,g)):
            assert not set(g.iloc[a])&set(g.iloc[b])
            t=time.perf_counter();model=fit(make_model(name),X.iloc[a],y.iloc[a])
            pred[b]=model.predict(X.iloc[b]);fold_index[b]=k
            tr=score(y.iloc[a],model.predict(X.iloc[a]));va=score(y.iloc[b],pred[b])
            row={'name':name,'seed':seed,'fold':k,'train_f1':tr,'validation_f1':va,'gap':tr-va,'iterations':int(np.max(model['clf'].n_iter_)),'seconds':time.perf_counter()-t}
            rows.append(row);local_rows.append(row)
        s={'name':name,'seed':seed,'oof_f1':score(y,pred),'fold_mean':float(np.mean([r['validation_f1'] for r in local_rows])),'fold_std':float(np.std([r['validation_f1'] for r in local_rows])),'train_mean':float(np.mean([r['train_f1'] for r in local_rows])),'gap':float(np.mean([r['gap'] for r in local_rows]))}
        summaries.append(s);all_oof[name,seed]=pred
        predictions.append(pd.DataFrame({'id':ids,'label':y,'name':name,'seed':seed,'fold':fold_index,'prediction':pred}))
        pd.DataFrame(rows).to_csv('reports/folds_008b.csv',index=False)
        pd.DataFrame(summaries).to_csv('reports/resultados_cv_008b.csv',index=False)
        print(s,flush=True)
        return s
    for seed in SEEDS:
        for name in NAMES:evaluate(name,seed)
    results=pd.DataFrame(summaries);fold_results=pd.DataFrame(rows);checks=[];eligible=[]
    for name in NAMES[1:]:
        deltas=[];gap_deltas=[];wins=0
        for seed in SEEDS:
            ref=results[(results.name=='ref007')&(results.seed==seed)].iloc[0]
            cand=results[(results.name==name)&(results.seed==seed)].iloc[0]
            rf=fold_results[(fold_results.name=='ref007')&(fold_results.seed==seed)].sort_values('fold')
            cf=fold_results[(fold_results.name==name)&(fold_results.seed==seed)].sort_values('fold')
            wins+=int(np.count_nonzero(cf.validation_f1.to_numpy()>rf.validation_f1.to_numpy()))
            deltas.append(float(cand.oof_f1-ref.oof_f1));gap_deltas.append(float(cand.gap-ref.gap))
        passed=bool(wins>=7 and np.mean(deltas)>=0.003 and np.mean(gap_deltas)<=0)
        checks.append({'name':name,'mean_f1_gain':float(np.mean(deltas)),'folds_won':wins,'mean_gap_change':float(np.mean(gap_deltas)),'passes':passed})
        if passed:eligible.append(name)
    pd.DataFrame(checks).to_csv('reports/criterios_008b.csv',index=False)
    historical={};accepted=[]
    if eligible:
        ref=evaluate('ref007',137)
        for name in eligible:
            cand=evaluate(name,137);historical[name]={'candidate':cand['oof_f1'],'reference':ref['oof_f1'],'passes':cand['oof_f1']>=ref['oof_f1']}
            if historical[name]['passes']:accepted.append(name)
    pd.concat(predictions,ignore_index=True).to_csv('reports/predicciones_oof_008b.csv',index=False)
    winner=None
    if accepted:
        ordering=results[results.name.isin(accepted)].groupby('name').agg(f1=('oof_f1','mean'),gap=('gap','mean')).sort_values(['f1','gap'],ascending=[False,True])
        winner=str(ordering.index[0])
    rec={'plan':plan,'winner':winner,'submission_created':False,'criteria':checks,'historical':historical,'seed_results':summaries,'reference_public007':0.88888,'public006':'aproximadamente0.87 según documento del auditor; no se aportó valor exacto/captura en este turno','public008':None,'input_sha256':inputs,'versions':json.loads(Path('reports/registro_005.json').read_text())['versions'],'local_test_reused':False,'evaluation_type':'CV repetida usada para selección; no evaluación externa independiente'}
    if winner:
        model=fit(make_model(winner),train.text.fillna(''),train.label)
        joblib.dump(model,'models/modelo_008.joblib')
        ev=pd.read_csv('data/eval.csv',dtype={'id':str});sample=pd.read_csv('data/sample_submission.csv',dtype={'id':str})
        assert ev.id.is_unique and sample.id.is_unique and set(ev.id)==set(sample.id)
        p=model.predict(ev.text.fillna(''))
        sub=pd.DataFrame({'id':sample.id,'answer':sample.id.map(pd.Series(p,index=ev.id))})
        assert len(sub)==len(ev)==len(sample) and sub.id.is_unique and sub.id.equals(sample.id)
        assert not sub.isna().any().any() and set(sub.answer)<=set(LABELS)
        sub.to_csv('submissions/submission_008.csv',index=False)
        pd.testing.assert_frame_equal(sub,pd.read_csv('submissions/submission_008.csv',dtype={'id':str}))
        assert np.array_equal(joblib.load('models/modelo_008.joblib').predict(ev.text.fillna('')),p)
        rec.update(submission_created=True,artifacts_sha256={p:sha(p) for p in ['models/modelo_008.joblib','submissions/submission_008.csv']},csv_checks_passed=True)
    else:
        rec['decision']='Conservar007: ninguna candidata cumple todos los criterios previos. No generar un CSV equivalente ni reducir los umbrales después de observar resultados.'
    assert all(sha(p)==v for p,v in protected.items())
    rec['old_artifacts_unchanged']=True;rec['elapsed_seconds']=time.perf_counter()-started
    rec['source_sha256']={p:sha(p) for p in ['src/experimento_008b.py','src/modelos_008b.py','src/features_007b.py','reports/particiones.csv']}
    dump('reports/registro_008b.json',rec)
    print(json.dumps({'winner':winner,'submission_created':rec['submission_created'],'decision':rec.get('decision'),'seconds':rec['elapsed_seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':run()
