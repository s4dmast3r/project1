"""Comprueba trazabilidad, predicciones OOF y exportación de011b."""
from src.experimento import sha,LABELS
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score,f1_score
r=json.loads(Path('reports/registro_011b.json').read_text())
for p,h in {**r['input_sha256'],**r['source_sha256'],**r.get('artifacts_sha256',{})}.items():assert sha(p)==h,p
old=json.loads(Path('reports/registro_013.json').read_text())
for p,h in {**old['source_sha256'],**old['artifacts_sha256']}.items():assert sha(p)==h,p
cv=pd.read_csv('reports/resultados_011b.csv');folds=pd.read_csv('reports/folds_011b.csv')
oof=pd.read_csv('reports/predicciones_oof_011b.csv',dtype={'id':str})
parts=pd.read_csv('reports/particiones.csv',dtype={'id':str});dev=set(parts.loc[parts.partition.eq('development'),'id'])
assert len(cv)==18+(1+len(r['historical']) if r['historical'] else 0)
assert len(folds)==len(cv)*3 and len(oof)==len(cv)*9600 and not folds.isna().any().any()
for (name,seed),p in oof.groupby(['name','seed']):
    assert p.id.is_unique and set(p.id)==dev
    row=cv[(cv.name==name)&(cv.seed==seed)].iloc[0]
    assert abs(accuracy_score(p.label,p.prediction)-row.accuracy)<1e-12
    assert abs(f1_score(p.label,p.prediction,labels=LABELS,average='macro')-row.f1)<1e-12
    for fold,subset in p.groupby('fold'):
        f=folds[(folds.name==name)&(folds.seed==seed)&(folds.fold==fold)].iloc[0]
        assert abs(accuracy_score(subset.label,subset.prediction)-f.validation_accuracy)<1e-12
        assert abs(f1_score(subset.label,subset.prediction,labels=LABELS,average='macro')-f.validation_f1)<1e-12
prev=pd.read_csv('reports/resultados_013.csv');prev=prev[prev.name=='l1_c2'].set_index('seed')
for _,row in cv[(cv.name=='ref010')&cv.seed.isin(r['plan']['seeds'])].iterrows():
    assert abs(row.accuracy-prev.loc[row.seed,'cand_acc'])<1e-12
    assert abs(row.f1-prev.loc[row.seed,'f1_cand'])<1e-12
check={'pipeline_fits_cv':len(folds),'oof_metrics_recomputed':len(cv),'only_development_in_oof':True,'hashes_match':True,'reference_matches_archived010':True,'submission_created':r['submission_created']}
if r['submission_created']:
    ev=pd.read_csv('data/eval.csv',dtype={'id':str});sample=pd.read_csv('data/sample_submission.csv',dtype={'id':str});sub=pd.read_csv('submissions/submission_011.csv',dtype={'id':str})
    assert sub.id.equals(sample.id) and sub.id.is_unique and len(sub)==3000 and list(sub.columns)==['id','answer']
    assert not sub.isna().any().any() and set(sub.answer)<=set(LABELS)
    model=joblib.load('models/modelo_011.joblib');pred=pd.Series(model.predict(ev.text),index=ev.id)
    assert np.array_equal(sub.answer,sub.id.map(pred))
    prior=pd.read_csv('submissions/submission_010.csv',dtype={'id':str});assert prior.id.equals(sub.id)
    check.update(csv_rows=len(sub),fresh_reload_matches_csv=True,changes_vs010=int((sub.answer!=prior.answer).sum()),class_counts=sub.answer.value_counts().to_dict())
else:
    assert not Path('submissions/submission_011.csv').exists() and not Path('models/modelo_011.joblib').exists()
Path('reports/verificacion_011b.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n');print(json.dumps(check,ensure_ascii=False,indent=2))
