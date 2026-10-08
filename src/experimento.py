"""Experimento reproducible: ejecutar desde la raíz del proyecto."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
import hashlib, json, platform, warnings
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

SEED = 42
LABELS = ['negativo', 'neutral', 'positivo']

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def normalized(s):
    return s.fillna('').str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()

def run():
    warnings.filterwarnings('error', category=ConvergenceWarning)
    inputs = {str(p): sha(p) for p in sorted(Path('data').glob('*.csv'))}
    train = pd.read_csv('data/train.csv', dtype={'id': str})
    ev = pd.read_csv('data/eval.csv', dtype={'id': str})
    sample = pd.read_csv('data/sample_submission.csv', dtype={'id': str})
    assert list(train) == ['id','text','label'] and list(ev) == ['id','text'] and list(sample) == ['id','answer']
    assert train.label.notna().all() and set(train.label) == set(LABELS)
    for df in [train, ev, sample]:
        assert df.id.notna().all() and df.id.is_unique
    assert set(ev.id) == set(sample.id) and len(ev) == len(sample)
    audit = {}
    for name, df in [('train', train), ('eval', ev), ('sample_submission', sample)]:
        a = {'shape': list(df.shape), 'dtypes': df.dtypes.astype(str).to_dict(), 'missing': df.isna().sum().to_dict(), 'unique_ids': df.id.is_unique}
        if 'text' in df:
            n = normalized(df.text)
            a.update(empty_text=int(n.eq('').sum()), exact_duplicate_rows=int(df.text.fillna('').duplicated().sum()), normalized_duplicate_rows=int(n.duplicated().sum()), repeated_groups=int(n.value_counts().gt(1).sum()))
        audit[name] = a
    groups = normalized(train.text)
    conflicts = train.assign(group=groups).groupby('group').label.nunique()
    audit['contradictory_groups_normalized'] = int(conflicts.gt(1).sum())
    audit['contradictory_groups_exact'] = int(train.groupby(train.text.fillna('')).label.nunique().gt(1).sum())
    audit['train_eval_overlap_exact'] = len(set(train.text.fillna('')) & set(ev.text.fillna('')))
    audit['train_eval_overlap_normalized'] = len(set(groups) & set(normalized(ev.text)))
    audit['class_counts'] = train.label.value_counts().to_dict()
    dump('reports/auditoria_datos.json', audit)
    X, y = train.text.fillna(''), train.label
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    dev, test = next(outer.split(X, y, groups))
    assert not set(groups.iloc[dev]) & set(groups.iloc[test])
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(inner.split(X.iloc[dev], y.iloc[dev], groups.iloc[dev]))
    partition = train[['id','label']].copy()
    partition['group_sha256'] = groups.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    partition['partition'] = 'local_test'
    partition['cv_fold'] = -1
    partition.loc[dev, 'partition'] = 'development'
    for i, (a,b) in enumerate(folds):
        assert not set(groups.iloc[dev[a]]) & set(groups.iloc[dev[b]])
        partition.loc[dev[b], 'cv_fold'] = i
    partition.to_csv('reports/particiones.csv', index=False)
    print('Auditoría:', json.dumps(audit, ensure_ascii=False, default=str), flush=True)
    print('Particiones:', partition.groupby(['partition','label']).size().to_dict(), flush=True)
    scoring = {'accuracy':'accuracy', 'f1_macro':'f1_macro'}
    dummy = cross_validate(DummyClassifier(strategy='most_frequent'), X.iloc[dev].to_numpy().reshape(-1,1), y.iloc[dev], cv=folds, scoring=scoring)
    pipe = Pipeline([('tfidf', TfidfVectorizer(lowercase=True, stop_words=None, strip_accents=None, norm='l2')), ('clf', LogisticRegression(solver='lbfgs', max_iter=2500, class_weight=None, random_state=SEED))])
    grid = {'tfidf__ngram_range':[(1,1),(1,2)], 'tfidf__min_df':[1,2], 'tfidf__sublinear_tf':[False,True], 'clf__C':[0.5,1.,2.,4.]}
    # Dos procesos, un hilo numérico por proceso; error_score evita ocultar fallos.
    search = GridSearchCV(pipe, grid, cv=folds, scoring=scoring, refit=False, n_jobs=2, pre_dispatch=2, error_score='raise', verbose=1)
    with joblib.parallel_backend('loky', inner_max_num_threads=1):
        search.fit(X.iloc[dev], y.iloc[dev])
    results = pd.DataFrame(search.cv_results_)
    results.to_csv('reports/resultados_cv.csv', index=False)
    # Criterio fijado antes de ver resultados: diferencia <= 0.001 en accuracy.
    eligible = results[results.mean_test_accuracy >= results.mean_test_accuracy.max()-0.001].copy()
    eligible['complexity'] = eligible.params.map(lambda p: p['tfidf__ngram_range'][1])
    selected = eligible.sort_values(['complexity','param_tfidf__min_df','param_clf__C','param_tfidf__sublinear_tf'], ascending=[True,False,True,True]).iloc[0]
    params = selected['params']
    model = pipe.set_params(**params).fit(X.iloc[dev], y.iloc[dev])
    pred = model.predict(X.iloc[test])  # Única evaluación de la configuración elegida.
    baseline = DummyClassifier(strategy='most_frequent').fit(np.zeros((len(dev),1)), y.iloc[dev]).predict(np.zeros((len(test),1)))
    cm = confusion_matrix(y.iloc[test], pred, labels=LABELS)
    metrics = {'labels':LABELS, 'accuracy':accuracy_score(y.iloc[test], pred), 'f1_macro':f1_score(y.iloc[test], pred, average='macro'), 'per_class':classification_report(y.iloc[test], pred, labels=LABELS, output_dict=True, zero_division=0), 'confusion_counts':cm.tolist(), 'confusion_normalized':(cm/cm.sum(axis=1,keepdims=True)).tolist(), 'dummy_local':{'accuracy':accuracy_score(y.iloc[test], baseline), 'f1_macro':f1_score(y.iloc[test], baseline, average='macro')}, 'cv':{k:float(selected[k]) for k in ['mean_test_accuracy','std_test_accuracy','mean_test_f1_macro','std_test_f1_macro']}, 'dummy_cv':{k:float(v) for metric in scoring for k,v in [(metric+'_mean',np.mean(dummy['test_'+metric])),(metric+'_std',np.std(dummy['test_'+metric]))]}, 'selected_params':params, 'max_iterations_used':model['clf'].n_iter_.tolist()}
    dump('reports/evaluacion_local.json', metrics)
    cases = train.iloc[test].copy()
    cases['prediction'] = pred
    cases['correct'] = cases.label.eq(cases.prediction)
    cases['negation_marker'] = cases.text.fillna('').str.contains(r'\b(?:no|ni|nunca|sin)\b', case=False, regex=True)
    cases['contrast_marker'] = cases.text.fillna('').str.contains(r'\b(?:pero|aunque|sin embargo|al final)\b', case=False, regex=True)
    cases.to_csv('reports/predicciones_locales.csv',index=False)
    subsets = {'aciertos':cases.correct, 'errores':~cases.correct, 'confusiones_neutral':~cases.correct & (cases.label.eq('neutral') | cases.prediction.eq('neutral')), 'negaciones':cases.negation_marker, 'contraste_posible_mixto':cases.contrast_marker}
    examples = []
    for category, mask in subsets.items():
        subset = cases[mask]
        if len(subset):
            examples.append(subset.sample(n=min(4,len(subset)),random_state=SEED).assign(category=category))
    pd.concat(examples).to_csv('reports/ejemplos.csv',index=False)
    terms = model['tfidf'].get_feature_names_out()
    records = []
    for label, coef in zip(model.classes_, model['clf'].coef_):
        for kind, mask in [('unigrama', np.array([' ' not in t for t in terms])),('bigrama', np.array([' ' in t for t in terms]))]:
            idx = np.flatnonzero(mask)
            for direction, indices in [('positivo',idx[np.argsort(coef[idx])[-12:][::-1]]),('negativo',idx[np.argsort(coef[idx])[:12]])]:
                records.extend({'clase':label,'tipo':kind,'direccion':direction,'termino':terms[i],'coeficiente':float(coef[i])} for i in indices)
    pd.DataFrame(records).to_csv('reports/coeficientes.csv',index=False)
    # Tras cerrar la evaluación local, reentrenar exactamente la misma configuración.
    model.fit(X,y)
    joblib.dump(model, 'models/modelo_001.joblib')
    predicted = model.predict(ev.text.fillna(''))
    lookup = pd.Series(predicted,index=ev.id)
    submission = pd.DataFrame({'id':sample.id,'answer':sample.id.map(lookup)})
    assert len(submission)==len(ev) and submission.id.is_unique
    assert submission.id.equals(sample.id) and not submission.isna().any().any()
    assert set(submission.answer) <= set(LABELS)
    submission.to_csv('submissions/submission_001.csv',index=False)
    reread = pd.read_csv('submissions/submission_001.csv',dtype={'id':str})
    pd.testing.assert_frame_equal(reread,submission)
    reloaded = joblib.load('models/modelo_001.joblib')
    assert np.array_equal(reloaded.predict(ev.text.fillna('')),predicted)
    assert inputs == {p:sha(p) for p in inputs}
    record = {'seed':SEED,'input_sha256':inputs,'versions':{p:version(p) for p in ['scikit-learn','pandas','numpy','joblib','matplotlib','nbformat','nbclient','ipykernel']},'python':platform.python_version(),'selected_params':params,'selection_rule':'accuracy CV máxima; tolerancia absoluta 0.001; preferir unigramas, min_df mayor, C menor, sublinear_tf False, en ese orden','n_configurations':len(results),'development_rows':len(dev),'local_test_rows':len(test),'folds':5,'split_method':'primer fold StratifiedGroupKFold(5, shuffle=True, random_state=42); grupos: minúsculas y espacios normalizados; ninguna fila eliminada','metrics':metrics,'final_max_iterations_used':model['clf'].n_iter_.tolist(),'artifacts_sha256':{p:sha(p) for p in ['models/modelo_001.joblib','submissions/submission_001.csv','reports/particiones.csv']},'checks':{'input_files_unchanged':True,'group_disjointness':True,'submission_schema_order_ids_labels':True,'model_reload_identical_predictions':True,'csv_roundtrip':True},'kaggle_score':None}
    dump('reports/registro_experimento_001.json',record)
    print(json.dumps(metrics,ensure_ascii=False,indent=2),flush=True)
    return record

if __name__ == '__main__':
    run()
