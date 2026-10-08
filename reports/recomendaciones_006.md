# Verificación de submission_005 y recomendaciones para la versión 006

Fecha: 8 de octubre de 2026. Métrica principal exclusiva: F1 macro con etiquetas
explícitas y `zero_division=0`. Las recomendaciones se sustentan en las diapositivas
del curso (`slidesP1/`) y en evidencia OOF calculada localmente; ninguna cifra se
inventó y no se accedió a Kaggle.

## 1. Verificación de submission_005 — resultado: correcta

Comprobaciones realizadas sobre los artefactos reales:

- **Huellas**: SHA-256 de `models/modelo_005.joblib`, `submissions/submission_005.csv`,
  `src/experimento_005.py`, `src/features_004.py` coinciden con `registro_005.json`;
  los CSV originales siguen intactos (`ae686726…`, `c9aedcb9…`, `48752890…`).
- **CSV**: 3000 filas, columnas exactas `id,answer`, IDs únicos, mismo orden que
  `sample_submission.csv`, sin vacíos, etiquetas válidas (negativo 1064, positivo 1097,
  neutral 839), lectura de ida y vuelta sin cambios.
- **Modelo**: Pipeline `features` (TextFeatures `max_ngram=1, min_df=3, char_weight=0.0,
  max_features=160000`) + `LinearSVC(C=2, n_iter=68)`. La recarga del joblib reproduce
  exactamente las predicciones del CSV. Cambia 144 de 3000 predicciones frente al 004.
- **Protocolo**: CV anidada 3×3 (semillas 42/137); la selección de configuración ocurre
  **solo dentro del entrenamiento de cada fold externo** (corrige la debilidad del 004,
  que seleccionó sobre el agregado OOF); regla de selección y presupuesto fijados antes
  de ejecutar en `plan_005.json`; convergencia exigida (`CheckedLinearSVC`); la antigua
  prueba local no se reutiliza; artefactos previos preservados y verificados.
- **Coherencia de métricas**: CV de selección 0.874163; OOF anidada 0.873382; media de
  folds 0.873373 ± 0.004074; referencia 004 en los mismos folds 0.850578 (+0.0228);
  brecha 0.125293 (004: 0.148529). Por clase: negativo 0.819, neutral 0.988, positivo
  0.813. Todo coherente con los CSV intermedios (`resultados_cv_005.csv`,
  `cv_005_anidada.csv`, `predicciones_oof_005.csv`).

**Veredicto**: la 005 es una simplificación válida y verificada (solo unigramas
posicionales, sin canal de caracteres) que mejora la estimación externa local y reduce
la brecha. El entrenamiento sigue en 0.99946: la brecha persiste y no hay prueba virgen.
El público 005 está pendiente; el usuario decide el envío.

## 2. Qué aportan las diapositivas del curso

| Técnica del curso | Diapositiva | Aplicabilidad aquí | Riesgo |
|---|---|---|---|
| Umbral de decisión (no 0.5) | S10 (pág. 21-22), S9 | Medido: ganancia ≈ 0 (ver §3). Descartado. | Bajo, pero nulo |
| Ensambles por votación | S13 | Voto duro/blando directo perjudica (modelos débiles arrastran al fuerte). Solo con stacking que aprenda pesos. | Medio |
| Stacking (modelo de agregación) | S13 (pág. 18) | **Prometedor**: +0.005–0.007 OOF medido (§3). Meta-modelo simple y scores OOF honestos. | Bajo si se hace bien |
| Bagging / Random Forest | S13 | Posible miembro del stack; los árboles sobre TF-IDF suelen rendir menos que los lineales aquí. | Medio (poda: S12) |
| Boosting | S13 | «Se puede sobreajustar» (S13); exige validación de n_estimators. No es primera opción. | Alto si no se limita |
| Regularización Ridge/Lasso (C, L1) | S7/S8/S9 | Continuar la tendencia 003→004→005 de simplificar capacidad (C, min_df, quitar flag de epílogo). | Bajo |
| Preprocesamiento: vacías y stemming/lematización | S5 (pág. 8-14), S11-Caso | Ablación acotada; **preservar negaciones** («no me gustó» ≠ «me gustó», S11 pág. 9). Stemming Snowball-es con nltk (declararlo). | Bajo–medio |
| KNN con coseno | S12 | Complementario a lineales, pero «maldición de la dimensionalidad» (S12) exige vocabulario reducido. | Medio |
| SMOTE / ADASYN | S12/S13 | No aplica: desbalance leve (4212/4212/3576), «solo datos numéricos», «puede llevar a sobreajuste». | Alto, sin beneficio esperado |
| Embeddings (Word2vec/GloVe) | S5 (pág. 41-45) | **Prohibidos en esta etapa** (redes/embeddings preentrenados). | — |

## 3. Evidencia empírica recogida (OOF, mismos 3 folds externos semilla 137)

- **Umbral δ entre positivo/negativo** sobre los scores del LinearSVC (selección anidada
  por fold, S10): F1 por fold 0.8736 / 0.8679 / 0.8799 → media 0.8738, sin mejora real
  frente a 0.8734. La confusión neg↔pos es simétrica; no hay sesgo que corregir.
- **Modelos base** (características ng1 de 005): LinearSVC C=2 → 0.8734; LogisticRegression
  C=4 → 0.8599; MultinomialNB → 0.8324. En espacio de palabras 1-2 gramos (sin posición):
  LR → 0.7354, NB → 0.7159. Las características posicionales son las que aportan la señal.
- **Complementariedad** (fracción de los 1269 errores de 005 que cada modelo también
  falla): lr_ng1 89,8 % (muy parecido); nb_ng1 73,3 %; lr_word 68,2 %; nb_word 66,8 %.
  Los modelos de **otro espacio de características** fallan distinto → útiles para stacking.
- **Stacking honesto** (meta LogisticRegression C=1 sobre scores rankeados por fold,
  meta entrenada solo con folds ajenos; S13 «modelo para la agregación»):
  svc+lr_ng1 → 0.8775; svc+lr_ng1+nb_ng1 → 0.8788; **svc_ng1+lr_word+nb_word → 0.8805**
  (+0.0071 sobre 005). Ganancia modesta pero real, y el meta-modelo es de baja capacidad.

## 4. Plan priorizado para la ronda 006 (hipótesis, fijadas por escrito)

H1 — **Stacking con meta-modelo simple (S13)**. Bases: LinearSVC sobre ng1 (005),
LogisticRegression y MultinomialNB sobre palabras 1-2 gramos sin posición. Meta:
LogisticRegression C=1 sobre scores rankeados, entrenada únicamente con scores OOF de
folds ajenos. Expectativa: +0.005–0.007 OOF. Riesgo: bajo (meta de baja capacidad;
bases distintas en algoritmo y en espacio de características, que es la condición de
independencia que pide S13).

H2 — **Preprocesamiento clásico acotado (S5/S11)**. Ablaciones: (a) quitar palabras
vacías del corpus pero preservando negaciones y el léxico discursivo (lista propia,
no lexicón de polaridad); (b) stemming Snowball para español (nltk; declarar el recurso).
Dos o tres configuraciones, una ablación por eje. Expectativa: incierta, ±0.005; el
corpus es plantillado y las vacías son parte de las plantillas posicionales.

H3 — **Capacidad/regularización del ganador (S7/S8/S9)**. Sobre ng1: C ∈ {0.5, 1},
min_df ∈ {3, 5}, y quitar el flag `epilogue{0,1}` de los canales posicionales. Objetivo:
reducir la brecha 0.125 sin perder F1; si se gana F1, mejor.

H4 — **Diversidad adicional para el stack (S12/S13)**: RandomForest con poda
(max_depth ≤ 16) sobre las features ng1, y/o KNN coseno con vocabulario recortado
(max_features ≈ 3000). Solo si H1-H3 dejan presupuesto; cada modelo se incorpora si su
error OOF solapa < 85 % de los errores del stack actual.

**No recomendado**: SMOTE/ADASYN (S12/S13: desbalance leve, solo numérico, riesgo de
sobreajuste), umbrales de decisión (ganancia medida ≈ 0), boosting agresivo
(S13: sobreajusta si no se limita), votación blanda directa (los modelos débiles
arrastran al fuerte; medido), embeddings (prohibidos).

## 5. Protocolo de evaluación para 006 (si el usuario aprueba ejecutar)

- Mismos 3 folds externos (semilla 137, reconstruidos de `predicciones_oof_005.csv`)
  para comparación emparejada con 005; selección de configuración **solo dentro del
  entrenamiento de cada fold** (como 005, no como 004).
- Métrica: F1 macro exclusivamente. Regla: elegibles a ≤0.003 del máximo; entre ellos
  menor brecha; la ganadora reemplaza 005 solo si supera su OOF en ≥0.002. Si no, se
  conserva 005 y se documenta el aprendizaje.
- Stacking: bases reajustadas por fold; meta solo con scores OOF (nunca in-sample).
  Producción: bases entrenadas con todo train, meta entrenada con los scores OOF de
  desarrollo; verificación completa del CSV (esquema, orden, etiquetas, recarga).
- Los folds externos ya fueron examinados en 003–005: la OOF es comparación emparejada,
  no prueba virgen; la adaptación acumulada se declara y no se presenta como
  confirmación independiente.

## 6. Resultados públicos y estado

- Público 002: 0.73888; 003: 0.84333; 004: **0.86111** (capturas; nombre de métrica no
  visible; el usuario sube manualmente). Público 005: pendiente.
- La OOF local subestimó el público en ~0.01 en 003 y 004 (0.8354→0.84333 y
  0.8506→0.86111): no extrapolar ni prometer valores públicos.
- El 0,90 no se alcanzó en ninguna evaluación local honesta (máximo actual: OOF 0.8805
  del stacking exploratorio). El camino más sólido documentado hacia arriba es H1+H2+H3.
