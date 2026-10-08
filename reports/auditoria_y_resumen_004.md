# Auditoría técnica y versión 004 — F1 macro

Fecha: 8 de octubre de 2026. Auditoría realizada sobre la evidencia real del
proyecto; ninguna cifra se inventó y no se accedió a Kaggle. Métrica principal
exclusiva: F1 macro con etiquetas explícitas y `zero_division=0`.

## 1. Integridad y reproducibilidad (verificado)

- SHA-256 de `data/*.csv` coinciden con los tres registros históricos
  (train `ae686726…`, eval `c9aedcb9…`, sample `48752890…`): los CSV originales
  siguen intactos.
- SHA-256 de `models/modelo_001/002/003.joblib`, `submissions/submission_001/002/003.csv`
  y `reports/particiones.csv` coinciden con sus registros.
- Entorno verificado en `.venv`: Python 3.14.7, scikit-learn 1.9.1, pandas 3.0.6,
  numpy 2.5.3, joblib 1.6.0 (igual al registrado).
- `submission_003.csv` re-verificado: 3000 filas, columnas `id,answer`, IDs únicos
  en el orden de `sample_submission.csv`, sin vacíos, etiquetas válidas, y la
  recarga de `modelo_003.joblib` reproduce las predicciones.

## 2. Hallazgos de auditoría (severidad / evidencia / impacto)

1. **Sobreajuste severo del 003 (confirmado, no hipotético).** `resultados_cv_003.csv`
   y `registro_experimento_003.json`: F1 macro de entrenamiento 0.999951 frente a
   0.833779 en CV (brecha 0.166171). El modelo memoriza casi perfectamente el
   desarrollo. La CV anidada da OOF 0.835400, honesta como estimación, pero la brecha
   no se resuelve solo cambiando el estimador.
2. **Cuello de botella positivo/negativo (confirmado).** OOF por clase del 003:
   negativo 0.761, neutral 0.989, positivo 0.756. Confusiones casi simétricas
   (neg→pos 785, pos→neg 801). Las reseñas mezclan aspectos positivos y negativos
   (ejemplos reproducibles en `predicciones_oof_003.csv`); el modelo pesa mal los
   aspectos en conflicto.
3. **Corpus fuertemente plantillado (riesgo, documentado).** Hay plantillas de
   apertura («Lo pedí a mediados de mes…», «Aporto mi granito de arena») y de cierre
   («Cada quien que juzgue», «En fin, ni fu ni fa», «Para gustos, colores»). Los
   marcadores `EPILOGUE` de `features_003.py` corresponden a esas plantillas de
   cierre. La distribución por etiqueta de esas frases **no** es determinante (por
   ejemplo «Cada quien que juzgue»: 69 positivo / 58 negativo), así que no se
   demostró un atajo limpio por plantilla, pero las características posicionales
   sí explotan esa estructura y exigen una ablación.
4. **Discurso ≈ posición.** En `resultados_cv_003.csv`, `discourse C=2` (0.833779) y
   `position C=2` (0.832819) difieren ~0.001; el mecanismo de cierre/conectores
   aporta poco sobre la posición simple. Se necesitaba una ablación (ejecutada en §4).
5. **Aviso de recurso.** `ejecucion_003.log` registra «A worker stopped while some
   jobs were given to the executor» (joblib/loky, 2 procesos). La ejecución completó
   todas las puntuaciones, pero el paralelismo estuvo cerca del límite de memoria.
6. **Sin fuga de información detectada (verificado).** `features_003.py` construye los
   vectorizadores dentro de `fit()` (vocabulario e IDF por fold, sin estado
   compartido); `experimento_003.py` clona el modelo base 002 y el mejor estimador
   (`clone(...)`), y los grupos se mantienen disjuntos en todos los niveles
   (`assert` en CV interna, externa y partición original).
7. **Cálculo de F1 macro correcto.** `make_scorer(f1_score, average='macro',
   labels=LABELS, zero_division=0)`; las métricas OOF agrupadas y por fold son
   coherentes (0.835400 agrupada vs 0.835396 media de folds), y la matriz de
   confusión del 003 suma correctamente por clase.
8. **Discrepancia de métrica (visibilidad).** El prompt inicial describía accuracy;
   el usuario cambió el criterio a F1 macro. Las capturas públicas solo muestran
   «Public Score» sin el nombre de la métrica; la métrica oficial no quedó verificada.

## 3. Plan priorizado (fijado antes de ejecutar)

Hipótesis ordenadas por utilidad esperada, coste y riesgo (una ablación por eje):

| # | Hipótesis | Eje | Razonamiento |
|---|---|---|---|
| H1 | Reducir n-gramas posicionales de (1–3) a (1–2) | `max_ngram` | Los trigramas posicionales son muy específicos y sospechosos de memorización. |
| H2 | Conservar el cierre (epílogo) en el cuerpo | `keep_epilogue` | Ablación pedida explícitamente en el prompt. |
| H3 | Segmentar solo por oraciones, sin conectores | `use_connector` | Mide cuánto aporta la segmentación por conectores. |
| H4 | Subir frecuencia mínima de 3 a 7 | `min_df` | Reduce características raras y capacidad. |
| H5 | Quitar el canal de caracteres | `char_weight` | El canal de caracteres puede añadir ruido/memorización. |
| H6 | Regularizar C de 2.0 a 0.5 | `C` | Reduce capacidad del SVM lineal. |

Protocolo: CV de selección 3-fold (StratifiedGroupKFold, semilla 42) sobre desarrollo;
OOF sobre los **mismos** 3 folds externos de 003 (semilla 137, reconstruidos desde
`predicciones_oof_003.csv`) para comparación emparejada. Criterio de elección: máximo
F1 macro OOF; entre configuraciones a ≤0.003 del máximo, menor brecha F1
entrenamiento-validación; desempate mayor OOF. Criterio de parada: la ganadora
reemplaza 003 solo si no es la réplica y supera su OOF en más de 0.002.

## 4. Ronda 004 ejecutada

Réplica `ref003` reproduce exactamente el 003 (CV 0.833779, OOF 0.835400), lo que
valida la reimplementación (`features_004.py` idéntica a `features_003.py` en sus
valores por defecto) y la reconstrucción de folds.

| Configuración | F1 entrenamiento (CV) | F1 CV selección | Brecha | F1 OOF externa |
|---|---:|---:|---:|---:|
| ref003 (003) | 0.999951 | 0.833779 | 0.166171 | 0.835400 |
| keep_epi | 1.000000 | 0.833436 | 0.166564 | 0.834634 |
| no_conn | 1.000000 | 0.823544 | 0.176456 | 0.822886 |
| **ng2 (ganadora)** | **0.999951** | **0.851422** | **0.148529** | **0.850578** |
| mindf7 | 0.999901 | 0.834832 | 0.165069 | 0.835783 |
| no_char | 0.999951 | 0.835140 | 0.164811 | 0.834573 |
| c05 | 0.997280 | 0.822944 | 0.174335 | 0.822382 |
| mindf7_no_char | 0.999555 | 0.839118 | 0.160437 | 0.837371 |

Lectura:

- **H1 (quitar trigramas posicionales) es la única mejora clara**: +0.0152 de F1 OOF
  sobre 003 (0.850578 vs 0.835400) y −0.0176 de brecha. Es una simplificación, no un
  añadido de complejidad: los trigramas con posición memorizaban plantillas específicas.
- **H2 (conservar el cierre)** no ayuda: keep_epi queda ligeramente por debajo
  (0.834634). Excluir el epílogo del cuerpo es inocuo o marginalmente beneficioso.
- **H3 (quitar conectores)** perjudica claramente (0.822886): la segmentación por
  conectores aporta ~0.013 de F1.
- **H4/H5/H6** no mejoran al 003 de forma relevante; `c05` empeora (0.822382), es
  decir, reducir C solo no compensa la especificidad de las características.
- `mindf7_no_char` (0.837371) no alcanza a `ng2`; la combinación no aporta.

Ganadora por la regla fijada: **`ng2`** (discourse, `max_ngram=2`, `min_df=3`,
`char_weight=0.5`, C=2.0), única config en la banda de 0.003. Supera a 003 en
0.015178 > 0.002, por lo que reemplaza a 003.

## 5. F1 de evaluación (OOF) y límites

| Clase | F1 003 | F1 004 | |
|---|---:|---:|---|
| negativo | 0.7614 | 0.7834 | +0.022 |
| neutral | 0.9886 | 0.9903 | +0.002 |
| positivo | 0.7562 | 0.7781 | +0.022 |
| **macro** | **0.8354** | **0.8506** | **+0.0152** |

Matriz de confusión 004 (OOF, filas=real, columnas=predicho, orden
negativo/neutral/positivo): `[[2645,16,709],[2,2854,5],[736,33,2600]]`. La mejora se
concentra en el cuello positivo/negativo (neg→pos baja de 785 a 709; pos→neg de 801
a 736) y neutral se mantiene.

**Límites (obligatorios):** la OOF se calcula sobre los mismos folds externos que ya
se observaron en 003; es una comparación emparejada, **no** una evaluación virgen.
No existe una nueva prueba etiquetada. No se puede descartar adaptación acumulada al
desarrollo ni dependencia por plantillas/origen no documentado. El 0,90 **no** se
alcanzó en ninguna evaluación local honesta (máximo OOF 0.8506).

## 6. Riesgo de sobreajuste

La brecha baja de 0.166 a 0.149, pero el entrenamiento sigue en 0.99995: el modelo
sigue memorizando el desarrollo casi por completo. La mejora de F1 OOF no elimina el
sobreajuste ni garantiza el comportamiento en el leaderboard privado. La decisión se
basó en OOF y en la simplicidad (menos características), no en el leaderboard.

## 7. Resultados públicos y pendientes

- 002: público 0.73888 (captura; nombre de métrica no visible).
- 003: público 0.84333 (captura; nombre de métrica no visible). Aumento público
  0.10445 en la escala del score; no se denomina mejora de F1 oficial.
- 004: público **pendiente**; el usuario decidirá y realizará el envío manual.

## 8. Decisión y entregables

Se recomienda `submissions/submission_004.csv` (verificado: 3000 filas, columnas
`id,answer`, IDs en el orden de `sample_submission.csv`, sin vacíos, etiquetas
válidas, lectura de ida y vuelta, y recarga de `models/modelo_004.joblib` que
reproduce las predicciones). Cambia 121 de 3000 predicciones respecto al 003.

Artefactos nuevos (los previos se conservan):

- `src/features_004.py`, `src/experimento_004.py`
- `reports/plan_004.json`, `reports/resultados_cv_004.csv`,
  `reports/predicciones_oof_004.csv`, `reports/registro_004.json`,
  `reports/ejecucion_004.log`
- `models/modelo_004.joblib`, `submissions/submission_004.csv`

El modelo 004 carga con `src/features_004.py` en el `PYTHONPATH` (ejecutar desde la
raíz). Reproducir: `.venv/bin/python -m src.experimento_004`.
