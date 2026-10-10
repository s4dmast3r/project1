# Versión 009 — regularización L1 sobre la representación 007

Fecha: 8 de octubre de 2026. Métrica exclusiva: F1 macro (`zero_division=0`).

## Cambio

Mismo espacio de características que la 007 (`TextFeatures007b(thirds=True)`:
unigramas/bigramas globales + unigramas posicionales por cláusula + cierre
discursivo + tercios), pero con clasificador **LinearSVC con regularización L1**
(`penalty='l1', dual=False, C=1, max_iter=10000, random_state=42`) en lugar del
L2 C=2 de la 007.

El efecto: L1 fuerza a cero la mayoría de los pesos (selección automática de
rasgos). El modelo final conserva solo **3252 coeficientes no nulos de 147645
(2,2 %)**. Es una simplificación real, no un añadido de capacidad.

## Por qué funcionó

La 007 tenía un sobreajuste extremo: F1 de entrenamiento 0.9995 y brecha 0.123
sobre 49.215 características. L1 colapsa el modelo a los rasgos informativos y
baja el entrenamiento a ~0.958, reduciendo la brecha a ~0.058 y mejorando la
generalización. Es el tradeoff sesgo-varianza del curso (S7/S8): el modelo estaba
en el lado de alta varianza.

## Evidencia (criterio fijado antes de ejecutar, `plan_009.json`)

Semillas nuevas 1009/2027/3053 (3 folds cada una) + puerta histórica 137:

| Semilla | Δ F1 OOF (l1_c1 vs ref007) | Folds ganados | Δ brecha |
|---|---:|---:|---:|
| 1009 | +0.02428 | 3/3 | −0.06566 |
| 2027 | +0.02348 | 3/3 | −0.06422 |
| 3053 | +0.02387 | 3/3 | −0.06695 |
| **137** | **0.89909 vs 0.87782** | — | — |

**9/9 folds ganados**, delta medio **+0.02388** (≥0.003 exigido), brecha media
**−0.06561**, y puerta histórica superada. Aprobada por las cuatro puertas
pre-registradas: se generó `submission_009.csv`.

## Métricas por clase (semilla 3053)

| Clase | 007 | 009 (l1) |
|---|---:|---:|
| negativo | 0.8246 | **0.8606** |
| neutral | 0.9882 | 0.9858 |
| positivo | 0.8215 | **0.8596** |
| macro | 0.8772 | **0.9020** |

La mejora se concentra exactamente en el cuello positivo/negativo (confusión
neg→pos baja de 585 a 463; pos→neg de 614 a 434). Es la primera vez que una
evaluación local honesta alcanza ~0.90.

## Otras candidatas probadas (descartadas)

- `charwb` (canal de caracteres con límites de palabra 2-6): neutro/levemente
  negativo (−0.0009 de media, 4/9 folds). No aporta.
- `nbsvm_word` (NB-SVM sobre palabras 1-2 gramos): −0.165. Confirma que el espacio
  de palabras sin posición es mucho más débil; NB-SVM no lo rescata.

## Verificaciones

- `submission_009.csv`: 3000 filas, `id,answer`, IDs en orden de sample, etiquetas
  válidas, sin vacíos, lectura ida y vuelta y recarga del Pipeline reproducida.
  Cambia 214 de 3000 predicciones respecto a 007.
- `models/modelo_009.joblib` (requiere `src/features_007b.py`); SHA-256 en
  `reports/registro_009.json`. Artefactos 001–008 intactos.
- Convergencia exigida (n_iter 1261 < 10000 en el ajuste final).

## Límites y honestidad

- La mejora local es grande, consistente (9/9 folds, 3 semillas nuevas + 137) y
  reduce el sobreajuste; es cualitativamente distinta del stacking de la 006
  (que fue +0.005 dentro del ruido y añadió complejidad).
- Aun así, no se puede **garantizar** el público: la métrica oficial no está
  verificada y el desarrollo se ha explorado en muchas rondas. El 006 enseñó que
  una mejora local no siempre transfiere, aunque aquí la señal es mucho mayor.
- Mejor público confirmado: 007 = 0.88888. El usuario decide y realiza el envío
  manual de la 009; el resultado se registrará con su captura.
