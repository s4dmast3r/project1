"""Ronda 011: re-ensayar bajo L1 rasgos que fallaron bajo L2.

La 010 aisló que la ganancia de 009 viene de la penalización L1 (el control L2 C=1
retrocede a brecha 0.12 y OOF ~0.873). L1 hace selección de rasgos, así que puede
aprovechar rasgos que bajo L2 solo añadían memorización. Candidatas (todas
LinearSVC penalty='l1', dual=False, C=1):

- ref009     : TextFeatures007b(thirds=True) — referencia exacta 009.
- l1_global13: añade trigramas al canal global (global_max_ngram=3); bajo L2 empeoró.
- l1_edgebig : añade bigramas posicionales en los bordes de cláusula; bajo L2 empeoró.
- l1_charwb  : añade canal de caracteres con límites de palabra (char_wb 2-6).
- l1_min2    : min_df=2 (más rasgos raros; L1 los poda).

Protocolo pre-registrado: semillas NUEVAS (2203, 3307, 4409) + puerta histórica 137.
Reemplaza 009 solo si cumple las cuatro puertas; si no, se conserva 009 (público
0.89555). No se bajan umbrales tras ver resultados.
"""
from src.experimento import normalized, sha, dump, LABELS
from src.features_007b import TextFeatures007b
from pathlib import Path
import time, warnings, json
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

FRESH_SEEDS = [2203, 3307, 4409]
HISTORICAL_SEED = 137
NAMES = ['ref009', 'l1_global13', 'l1_edgebig', 'l1_charwb', 'l1_min2']
REFERENCE_137_OOF = 0.8990913974695345


class CharWbFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.union_ = FeatureUnion([
            ('base', TextFeatures007b(thirds=True, min_df=3, max_features=160000)),
            ('charwb', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 6), min_df=3,
                                       sublinear_tf=True, max_features=70000)),
        ], transformer_weights={'base': 1.0, 'charwb': 0.5})
        self.union_.fit(X)
        return self

    def transform(self, X):
        return self.union_.transform(X)


def make_model(name):
    clf = LinearSVC(penalty='l1', dual=False, C=1, max_iter=10000, random_state=42)
    if name == 'ref009':
        feats = TextFeatures007b(thirds=True, min_df=3, max_features=160000)
    elif name == 'l1_global13':
        feats = TextFeatures007b(thirds=True, min_df=3, max_features=160000, global_max_ngram=3)
    elif name == 'l1_edgebig':
        feats = TextFeatures007b(thirds=True, min_df=3, max_features=160000, edge_bigrams=True)
    elif name == 'l1_charwb':
        feats = CharWbFeatures()
    elif name == 'l1_min2':
        feats = TextFeatures007b(thirds=True, min_df=2, max_features=160000)
    else:
        raise ValueError(name)
    return Pipeline([('features', feats), ('clf', clf)])


def fit(pipe, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        pipe.fit(X, y)
    assert np.max(pipe['clf'].n_iter_) < pipe['clf'].max_iter
    return pipe


def score(y, p):
    return float(f1_score(y, p, average='macro', labels=LABELS, zero_division=0))


def evaluate(X, y, g, name, seed):
    pred = np.empty(len(y), dtype=object)
    val_scores = []
    train_scores = []
    for k, (a, b) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(X, y, g)):
        assert not set(g.iloc[a]) & set(g.iloc[b])
        model = fit(make_model(name), X.iloc[a], y.iloc[a])
        pred[b] = model.predict(X.iloc[b])
        train_scores.append(score(y.iloc[a], model.predict(X.iloc[a])))
        val_scores.append(score(y.iloc[b], pred[b]))
    return {'name': name, 'seed': seed, 'oof_f1': score(y, pred),
            'fold_val': val_scores, 'fold_train': train_scores,
            'gap': float(np.mean([t - v for t, v in zip(train_scores, val_scores)])),
            'pred': pred}


def run():
    started = time.perf_counter()
    if Path('reports/registro_011.json').exists():
        raise FileExistsError('Conservar resultados 011 antes de repetir.')
    train = pd.read_csv('data/train.csv', dtype={'id': str})
    parts = pd.read_csv('reports/particiones.csv', dtype={'id': str})
    d = parts.partition.eq('development')
    gall = normalized(train.text)
    assert not set(gall[d]) & set(gall[~d])
    X = train.text.fillna('')[d].reset_index(drop=True)
    y = train.label[d].reset_index(drop=True)
    g = gall[d].reset_index(drop=True)

    plan = {
        'candidates': NAMES,
        'fresh_seeds': FRESH_SEEDS,
        'historical_seed': HISTORICAL_SEED,
        'metric': 'f1_macro; labels explícitos; zero_division=0',
        'gates': [
            'ganar >=7 de 9 folds emparejados frente a ref009 (3 semillas nuevas)',
            'delta medio F1 macro OOF >= 0.003',
            'brecha media <= brecha media de ref009',
            'no perder F1 OOF frente a ref009 en semilla histórica 137',
        ],
        'hypothesis': 'Bajo L1, rasgos que fallaron bajo L2 (trigramas globales, bigramas de borde, caracteres, rasgos raros) pueden ser podados selectivamente y aportar señal sin la memorización que los perjudicaba.',
        'limitations': 'Las semillas nuevas no crean datos vírgenes; todos los scores se usan para esta decisión pre-registrada, no son evaluación externa independiente.',
        'reference_public009': 0.89555,
    }
    dump('reports/plan_011.json', plan)

    seed_rows = []
    results = {}
    for seed in FRESH_SEEDS:
        ref = evaluate(X, y, g, 'ref009', seed)
        results['ref009', seed] = ref
        for name in NAMES[1:]:
            cand = evaluate(X, y, g, name, seed)
            results[name, seed] = cand
            deltas = np.array(cand['fold_val']) - np.array(ref['fold_val'])
            row = {'name': name, 'seed': seed, 'ref009_oof': ref['oof_f1'],
                   'cand_oof': cand['oof_f1'], 'oof_delta': cand['oof_f1'] - ref['oof_f1'],
                   'fold_wins': int((deltas > 0).sum()), 'gap_ref': ref['gap'],
                   'gap_cand': cand['gap'], 'gap_delta': cand['gap'] - ref['gap'],
                   'fold_deltas': np.round(deltas, 5).tolist()}
            seed_rows.append(row)
            print(row, flush=True)

    df = pd.DataFrame(seed_rows)
    agg = df.groupby('name').agg(oof_delta=('oof_delta', 'mean'),
                                 fold_wins=('fold_wins', 'sum'),
                                 gap_delta=('gap_delta', 'mean')).reset_index()
    checks = []
    eligible = []
    for name in NAMES[1:]:
        r = agg[agg.name == name].iloc[0]
        g1 = int(r.fold_wins) >= 7
        g2 = float(r.oof_delta) >= 0.003
        g3 = float(r.gap_delta) <= 0
        checks.append({'name': name, 'fold_wins': int(r.fold_wins),
                       'mean_oof_delta': float(r.oof_delta),
                       'mean_gap_delta': float(r.gap_delta),
                       'passes_fresh': bool(g1 and g2 and g3)})
        if g1 and g2 and g3:
            eligible.append(name)
    pd.DataFrame(seed_rows).to_csv('reports/resultados_011.csv', index=False)
    pd.DataFrame(checks).to_csv('reports/criterios_011.csv', index=False)

    hist = {}
    accepted = []
    if eligible:
        ref137 = evaluate(X, y, g, 'ref009', HISTORICAL_SEED)
        assert abs(ref137['oof_f1'] - REFERENCE_137_OOF) < 1e-9, (ref137['oof_f1'], REFERENCE_137_OOF)
        for name in eligible:
            cand = evaluate(X, y, g, name, HISTORICAL_SEED)
            hist[name] = {'ref009_137': ref137['oof_f1'], 'cand_137': cand['oof_f1'],
                          'passes': cand['oof_f1'] >= ref137['oof_f1']}
            if hist[name]['passes']:
                accepted.append(name)
            print('puerta 137', name, hist[name], flush=True)

    winner = None
    if accepted:
        winner = str(agg[agg.name.isin(accepted)].sort_values(
            ['oof_delta', 'gap_delta'], ascending=[False, True]).iloc[0]['name'])

    rec = {'plan': plan, 'seed_rows': seed_rows, 'checks': checks, 'historical_137': hist,
           'winner': winner, 'submission_created': False, 'reference_public009': 0.89555}

    if winner:
        model = fit(make_model(winner), train.text.fillna(''), train.label)
        joblib.dump(model, 'models/modelo_011.joblib')
        ev = pd.read_csv('data/eval.csv', dtype={'id': str})
        sample = pd.read_csv('data/sample_submission.csv', dtype={'id': str})
        assert ev.id.is_unique and sample.id.is_unique and set(ev.id) == set(sample.id)
        p = model.predict(ev.text.fillna(''))
        sub = pd.DataFrame({'id': sample.id, 'answer': sample.id.map(pd.Series(p, index=ev.id))})
        assert len(sub) == len(ev) == len(sample) and sub.id.is_unique and sub.id.equals(sample.id)
        assert not sub.isna().any().any() and set(sub.answer) <= set(LABELS)
        sub.to_csv('submissions/submission_011.csv', index=False)
        pd.testing.assert_frame_equal(sub, pd.read_csv('submissions/submission_011.csv', dtype={'id': str}))
        assert np.array_equal(joblib.load('models/modelo_011.joblib').predict(ev.text.fillna('')), p)
        pred_diag = evaluate(X, y, g, winner, FRESH_SEEDS[-1])['pred']
        per_class = classification_report(y, pred_diag, labels=LABELS, output_dict=True, zero_division=0)
        per_class.pop('accuracy', None)
        rec.update(submission_created=True,
                   artifacts_sha256={pth: sha(pth) for pth in ['models/modelo_011.joblib', 'submissions/submission_011.csv']},
                   per_class=per_class, confusion=confusion_matrix(y, pred_diag, labels=LABELS).tolist(),
                   csv_checks_passed=True,
                   decision=f'{winner} aprobada por las puertas pre-registradas; submission_011.csv generado.')
    else:
        rec['decision'] = 'Conservar009: ninguna candidata cumple las cuatro puertas pre-registradas. No generar CSV.'
    rec['elapsed_seconds'] = time.perf_counter() - started
    rec['source_sha256'] = {p: sha(p) for p in ['src/experimento_011.py', 'src/features_007b.py', 'reports/particiones.csv']}
    dump('reports/registro_011.json', rec)
    print(json.dumps({'winner': winner, 'submission_created': rec['submission_created'],
                      'decision': rec['decision'], 'seconds': rec['elapsed_seconds']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
