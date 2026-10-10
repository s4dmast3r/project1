"""Ronda 007b: variantes estructurales de 005; mismas puertas estrictas que 007.

Reutiliza el protocolo de la ronda 007 (tres semillas 271/599/913 + puerta
histórica 137) pero con cambios ESTRUCTURALES del espacio de características en
lugar de los canales de negación/contraste y regularización ya probados:
- global13  : canal global con trigramas (005 usa 1-2).
- mid_bucket: etiqueta 'mid' para palabras centrales de bloques largos.
- edge_bigrams: bigramas posicionales solo en los bordes de bloque.
- thirds    : etiqueta de tercio del texto completo para cada palabra.

Contraste previo (documentado en plan_007b.json): la hipótesis de "cláusula
dominante" con scorer de cláusulas de supervisión débil NO superó el contraste
(coincidencia con etiqueta 49,8 % global, 17,8 % en errores de 005), por lo que
no se construyen features de agregación por cláusula en esta ronda.

Todas las candidatas conservan el resto del diseño 005 (unigramas posicionales,
cierre discursivo, sin canal de caracteres, LinearSVC C=2). El criterio de
reemplazo exige +0.005 de F1 macro OOF en CADA semilla nueva y ganar 3 folds o 2
con menor brecha, y no perder en la semilla histórica 137.
"""
from src.experimento import normalized, sha, dump, LABELS
from src.features_007b import TextFeatures007b
from pathlib import Path
import time, warnings, json
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.exceptions import ConvergenceWarning

SEEDS = [271, 599, 913]
NAMES = ['ref005', 'global13', 'mid_bucket', 'edge_bigrams', 'thirds']
PARAMS = {
    'ref005': {},
    'global13': {'global_max_ngram': 3},
    'mid_bucket': {'mid_bucket': True},
    'edge_bigrams': {'edge_bigrams': True},
    'thirds': {'thirds': True},
}


def make_model(name):
    feats = TextFeatures007b(min_df=3, max_features=160000, **PARAMS[name])
    return Pipeline([('features', feats),
                     ('clf', LinearSVC(C=2, dual='auto', max_iter=10000, random_state=42))])


def fit(pipe, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        pipe.fit(X, y)
    assert np.max(pipe['clf'].n_iter_) < pipe['clf'].max_iter
    return pipe


def score(y, p):
    return float(f1_score(y, p, average='macro', labels=LABELS, zero_division=0))


def gates(oof_delta, fold_deltas, gap_delta):
    wins = int(np.count_nonzero(np.asarray(fold_deltas) > 0))
    return bool(oof_delta >= 0.005 and (wins == 3 or (wins >= 2 and gap_delta < 0)))


def run():
    started = time.perf_counter()
    if Path('reports/registro_007b.json').exists():
        raise FileExistsError('Conservar resultados 007b: archivar explícitamente antes de repetir.')
    old = json.loads(Path('reports/registro_005.json').read_text())
    for p, h in {**old['input_sha256'], **old['artifacts_sha256'], **old['source_sha256']}.items():
        assert sha(p) == h, p
    protected = {str(p): sha(p) for folder in ['data', 'models', 'submissions', 'src'] for p in Path(folder).glob('*') if p.is_file()}
    train = pd.read_csv('data/train.csv', dtype={'id': str})
    parts = pd.read_csv('reports/particiones.csv', dtype={'id': str})
    assert train.id.equals(parts.id)
    d = parts.partition.eq('development')
    gall = normalized(train.text)
    assert not set(gall[d]) & set(gall[~d])
    X = train.text.fillna('')[d].reset_index(drop=True)
    y = train.label[d].reset_index(drop=True)
    g = gall[d].reset_index(drop=True)
    ids = train.id[d].reset_index(drop=True)
    plan = {
        'seeds': SEEDS,
        'folds_per_seed': 3,
        'candidates': NAMES,
        'params': PARAMS,
        'metric': 'f1_macro; labels explícitos; zero_division=0',
        'selection': 'Cada semilla: delta F1 OOF >=0.005 vs005 y ganar 3 folds o >=2 con menor brecha media. Debe cumplir las tres semillas. Luego no perder F1 OOF en semilla histórica137. Entre aprobados mayor F1 medio en semillas nuevas; empate exacto menor brecha.',
        'historical_validation': '137 se usa como puerta de selección adicional, no prueba independiente.',
        'data_scope': '9600 filas de desarrollo histórico. Prueba local antigua excluida de selección.',
        'hypotheses': 'Cambios estructurales del espacio de características no probados en 003-007: trigramas globales, bucket medio, bigramas de borde, tercios del texto completo.',
        'clause_dominance_contrast': 'FALLO: scorer de cláusulas con supervisión débil (etiqueta de la reseña) coincide con la etiqueta en 49,8% del desarrollo y 17,8% de los errores 005; descartada la agregación por cláusula para esta ronda.',
        'excluded': 'neg_scope/post_contraste/c1_min5 ya probados en 007 sin pasar; stacking 006 cayó en público; sin EM, sin reetiquetado, sin datos externos.',
        'limitations': 'Cambiar semillas no crea datos vírgenes. 271 ya se usó dentro de stacking006. Todos estos scores se usan para selección, no son evaluación externa independiente.',
        'budget': '45 ajustes de screening; hasta 15 históricos condicionales; un ajuste final solo si aprobado',
        'n_jobs': 1,
    }
    dump('reports/plan_007b.json', plan)
    rows = []; summaries = []; predictions = []
    def evaluate(name, seed):
        pred = np.empty(len(y), dtype=object)
        fold_index = np.full(len(y), -1)
        local_rows = []
        for k, (a, b) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(X, y, g)):
            assert not set(g.iloc[a]) & set(g.iloc[b])
            t = time.perf_counter()
            model = fit(make_model(name), X.iloc[a], y.iloc[a])
            pred[b] = model.predict(X.iloc[b]); fold_index[b] = k
            tr = score(y.iloc[a], model.predict(X.iloc[a])); va = score(y.iloc[b], pred[b])
            row = {'name': name, 'seed': seed, 'fold': k, 'train_f1': tr, 'validation_f1': va,
                   'gap': tr - va, 'iterations': int(model['clf'].n_iter_), 'seconds': time.perf_counter() - t}
            rows.append(row); local_rows.append(row)
        s = {'name': name, 'seed': seed, 'oof_f1': score(y, pred),
             'fold_mean': float(np.mean([r['validation_f1'] for r in local_rows])),
             'fold_std': float(np.std([r['validation_f1'] for r in local_rows])),
             'train_mean': float(np.mean([r['train_f1'] for r in local_rows])),
             'gap': float(np.mean([r['gap'] for r in local_rows]))}
        summaries.append(s)
        predictions.append(pd.DataFrame({'id': ids, 'label': y, 'name': name, 'seed': seed,
                                         'fold': fold_index, 'prediction': pred}))
        pd.DataFrame(rows).to_csv('reports/folds_007b.csv', index=False)
        pd.DataFrame(summaries).to_csv('reports/resultados_cv_007b.csv', index=False)
        print(s, flush=True)
        return s
    for seed in SEEDS:
        for name in NAMES:
            evaluate(name, seed)
    results = pd.DataFrame(summaries); fold_results = pd.DataFrame(rows)
    checks = []; eligible = []
    for name in NAMES[1:]:
        passed = True
        for seed in SEEDS:
            ref = results[(results.name == 'ref005') & (results.seed == seed)].iloc[0]
            cand = results[(results.name == name) & (results.seed == seed)].iloc[0]
            ref_f = fold_results[(fold_results.name == 'ref005') & (fold_results.seed == seed)].sort_values('fold')
            c_f = fold_results[(fold_results.name == name) & (fold_results.seed == seed)].sort_values('fold')
            deltas = c_f.validation_f1.to_numpy() - ref_f.validation_f1.to_numpy()
            delta = float(cand.oof_f1 - ref.oof_f1); gd = float(cand.gap - ref.gap)
            ok = gates(delta, deltas, gd); passed = passed and ok
            checks.append({'name': name, 'seed': seed, 'f1_gain': delta, 'folds_won': int((deltas > 0).sum()),
                           'gap_change': gd, 'passes': ok})
        if passed:
            eligible.append(name)
    pd.DataFrame(checks).to_csv('reports/criterios_007b.csv', index=False)
    historical = {}; accepted = []
    if eligible:
        ref = evaluate('ref005', 137)
        for name in eligible:
            cand = evaluate(name, 137)
            historical[name] = {'candidate': cand['oof_f1'], 'reference': ref['oof_f1'],
                                'passes': cand['oof_f1'] >= ref['oof_f1']}
            if historical[name]['passes']:
                accepted.append(name)
    pd.concat(predictions, ignore_index=True).to_csv('reports/predicciones_oof_007b.csv', index=False)
    winner = None
    if accepted:
        ordering = results[results.name.isin(accepted)].groupby('name').agg(
            f1=('oof_f1', 'mean'), gap=('gap', 'mean')).sort_values(['f1', 'gap'], ascending=[False, True])
        winner = str(ordering.index[0])
    rec = {'plan': plan, 'winner': winner, 'submission_created': False, 'criteria': checks,
           'historical': historical, 'seed_results': summaries,
           'reference_public005': 0.88111,
           'public006': 'aproximadamente0.87 según documento del auditor; sin valor exacto/captura',
           'public007': None,
           'input_sha256': old['input_sha256'], 'versions': old['versions'],
           'local_test_reused': False,
           'evaluation_type': 'CV repetida usada para selección; no evaluación externa independiente'}
    if winner:
        model = fit(make_model(winner), train.text.fillna(''), train.label)
        joblib.dump(model, 'models/modelo_007.joblib')
        ev = pd.read_csv('data/eval.csv', dtype={'id': str})
        sample = pd.read_csv('data/sample_submission.csv', dtype={'id': str})
        assert ev.id.is_unique and sample.id.is_unique and set(ev.id) == set(sample.id)
        p = model.predict(ev.text.fillna(''))
        sub = pd.DataFrame({'id': sample.id, 'answer': sample.id.map(pd.Series(p, index=ev.id))})
        assert len(sub) == len(ev) == len(sample) and sub.id.is_unique and sub.id.equals(sample.id)
        assert not sub.isna().any().any() and set(sub.answer) <= set(LABELS)
        sub.to_csv('submissions/submission_007.csv', index=False)
        pd.testing.assert_frame_equal(sub, pd.read_csv('submissions/submission_007.csv', dtype={'id': str}))
        assert np.array_equal(joblib.load('models/modelo_007.joblib').predict(ev.text.fillna('')), p)
        rec.update(submission_created=True,
                   artifacts_sha256={p: sha(p) for p in ['models/modelo_007.joblib', 'submissions/submission_007.csv']},
                   csv_checks_passed=True)
    else:
        rec['decision'] = 'Conservar005: ninguna candidata cumple todos los criterios previos. No generar un CSV equivalente ni reducir los umbrales después de observar resultados.'
    assert all(sha(p) == v for p, v in protected.items())
    rec['old_artifacts_unchanged'] = True
    rec['elapsed_seconds'] = time.perf_counter() - started
    rec['source_sha256'] = {p: sha(p) for p in ['src/experimento_007b.py', 'src/features_007b.py', 'reports/particiones.csv']}
    dump('reports/registro_007b.json', rec)
    print(json.dumps({'winner': winner, 'submission_created': rec['submission_created'],
                      'decision': rec.get('decision'), 'seconds': rec['elapsed_seconds']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
