"""Corrección con selección solo en desarrollo; no reevalúa la prueba local."""
from src.experimento import normalized, sha, dump, SEED, LABELS
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.exceptions import ConvergenceWarning

RULE = ('Entre configuraciones a <=0.01 de la máxima accuracy CV, elegir la menor '
        'brecha media entrenamiento-validación (truncada en cero); desempatar por '
        'accuracy CV descendente. Regla fijada antes de ejecutar. No usar prueba local.')

def run():
    warnings.filterwarnings('error', category=ConvergenceWarning)
    old=json.loads(Path('reports/registro_experimento_001.json').read_text())
    protected={**old['input_sha256'],**old['artifacts_sha256']}
    assert all(sha(p)==v for p,v in protected.items())
    train=pd.read_csv('data/train.csv',dtype={'id':str})
    parts=pd.read_csv('reports/particiones.csv',dtype={'id':str})
    assert train.id.equals(parts.id)
    dev=np.flatnonzero(parts.partition.eq('development'))
    test=np.flatnonzero(parts.partition.eq('local_test'))
    X=train.text.fillna(''); y=train.label; groups=normalized(X)
    assert not set(groups.iloc[dev]) & set(groups.iloc[test])
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(X.iloc[dev],y.iloc[dev],groups.iloc[dev]))
    for i,(a,b) in enumerate(folds):
        assert set(dev[b])==set(np.flatnonzero(parts.cv_fold.eq(i)))
        assert not set(groups.iloc[dev[a]]) & set(groups.iloc[dev[b]])
    baseline=joblib.load('models/modelo_001.joblib')
    grid=[{'tfidf__ngram_range':[(1,1),(1,2)],'tfidf__min_df':[5],
           'tfidf__max_features':[1000,3000,10000], 'clf__C':[0.05,0.2,1.,4.]},
          {key:[value] for key,value in {**{k:tuple(v) if isinstance(v,list) else v for k,v in old['selected_params'].items()},'tfidf__max_features':None}.items()}]
    dump('reports/plan_regularizacion_002.json',{'rule':RULE,'grid':grid,'seed':SEED,'evaluation':'Solo CV desarrollo. Prueba local original ya vista, no se reevalúa ni se atribuye al modelo corregido.'})
    search=GridSearchCV(clone(baseline),grid,cv=folds,scoring={'accuracy':'accuracy','f1_macro':'f1_macro'},return_train_score=True,refit=False,n_jobs=2,pre_dispatch=2,error_score='raise',verbose=1)
    with joblib.parallel_backend('loky',inner_max_num_threads=1):
        search.fit(X.iloc[dev],y.iloc[dev])
    results=pd.DataFrame(search.cv_results_)
    results['accuracy_gap']=results.mean_train_accuracy-results.mean_test_accuracy
    results['selection_gap']=results.accuracy_gap.clip(lower=0)
    eligible=results[results.mean_test_accuracy>=results.mean_test_accuracy.max()-0.01]
    chosen=eligible.sort_values(['selection_gap','mean_test_accuracy'],ascending=[True,False]).iloc[0]
    results['selected']=results.index==chosen.name
    results.to_csv('reports/resultados_cv_regularizacion_002.csv',index=False)
    params=chosen['params']
    model=clone(baseline).set_params(**params)
    # Tras cerrar selección, todo train se utiliza únicamente para producción.
    model.fit(X,y)
    assert max(model['clf'].n_iter_)<model['clf'].max_iter
    joblib.dump(model,'models/modelo_002_regularizado.joblib')
    ev=pd.read_csv('data/eval.csv',dtype={'id':str})
    sample=pd.read_csv('data/sample_submission.csv',dtype={'id':str})
    assert ev.id.is_unique and sample.id.is_unique and set(ev.id)==set(sample.id)
    predictions=model.predict(ev.text.fillna(''))
    submission=pd.DataFrame({'id':sample.id,'answer':sample.id.map(pd.Series(predictions,index=ev.id))})
    assert len(submission)==len(ev)==len(sample) and submission.id.is_unique
    assert submission.id.equals(sample.id) and not submission.isna().any().any()
    assert set(submission.answer)<=set(LABELS)
    submission.to_csv('submissions/submission_002_regularizado.csv',index=False)
    pd.testing.assert_frame_equal(submission,pd.read_csv('submissions/submission_002_regularizado.csv',dtype={'id':str}))
    assert np.array_equal(joblib.load('models/modelo_002_regularizado.joblib').predict(ev.text.fillna('')),predictions)
    assert all(sha(p)==v for p,v in protected.items())
    reference=results.iloc[-1]
    metrics=['mean_train_accuracy','mean_test_accuracy','std_test_accuracy','mean_train_f1_macro','mean_test_f1_macro','std_test_f1_macro','accuracy_gap']
    record={'seed':SEED,'selection_rule':RULE,'configurations':len(results),'selected_params':params,
            'corrected_cv':{k:float(chosen[k]) for k in metrics},'original_cv_recomputed':{k:float(reference[k]) for k in metrics},
            'final_vocabulary_size':len(model['tfidf'].vocabulary_),'final_iterations':model['clf'].n_iter_.tolist(),
            'independent_test_metrics':None,'reason':'La prueba local del experimento original ya fue examinada; no se reevalúa. CV usada para selección no es una evaluación independiente.',
            'versions':old['versions'],'input_sha256':old['input_sha256'], 'originals_unchanged':True,
            'checks':{'group_disjointness':True,'csv_schema_ids_order_labels_roundtrip':True,'reloaded_predictions_identical':True},
            'artifact_sha256':{p:sha(p) for p in ['models/modelo_002_regularizado.joblib','submissions/submission_002_regularizado.csv','reports/particiones.csv']}}
    dump('reports/registro_experimento_002.json',record)
    summary=f'''# Corrección de sobreajuste

La regla se guardó antes de la búsqueda en plan_regularizacion_002.json: {RULE}

Se compararon {len(results)} configuraciones en los mismos cinco folds de desarrollo (9600 filas). TF-IDF permanece dentro del Pipeline. Se probaron vocabularios limitados, min_df=5 y distintos niveles de regularización. El modelo original participa como referencia. No se usa eval para ajustar vocabulario ni seleccionar modelos.

Configuración elegida: `{params}`.

| Accuracy | Original | Corrección |
|---|---:|---:|
| Entrenamiento (media folds) | {reference.mean_train_accuracy:.6f} | {chosen.mean_train_accuracy:.6f} |
| Validación (media folds) | {reference.mean_test_accuracy:.6f} | {chosen.mean_test_accuracy:.6f} |
| Brecha | {reference.accuracy_gap:.6f} | {chosen.accuracy_gap:.6f} |

F1 macro CV corregido: {chosen.mean_test_f1_macro:.6f} ± {chosen.std_test_f1_macro:.6f}. Accuracy CV: {chosen.mean_test_accuracy:.6f} ± {chosen.std_test_accuracy:.6f}.

La tolerancia de un punto porcentual limita la pérdida de validación permitida para favorecer una brecha menor. No se minimiza la brecha sin restricciones: un modelo que siempre predice la mayoría puede tener una brecha pequeña y ser inútil. La mejora de la brecha se refiere a estos folds; no demuestra ausencia absoluta de sobreajuste. La selección entre múltiples configuraciones puede introducir optimismo en CV, y reutiliza los folds del experimento anterior.

No se vuelve a evaluar la prueba local original: ya se observó y motivó esta revisión. Su accuracy anterior NO corresponde al modelo corregido. La evaluación independiente del nuevo modelo queda pendiente. No se garantiza un puntaje en Kaggle y no se realizó ningún envío.

Tras cerrar parámetros, se ajustó sobre todo train para producción. El CSV nuevo tiene 3000 filas con IDs en el orden requerido y predicciones verificadas tras recargar el modelo. Los originales se conservan. Archivo recomendado para enviar manualmente: submissions/submission_002_regularizado.csv. El número identifica una versión local, no un envío efectuado.
'''
    Path('reports/correccion_sobreajuste_002.md').write_text(summary)
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
    return record

if __name__=='__main__': run()
