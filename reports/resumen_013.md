# Versión 010 — optimización para accuracy (métrica oficial confirmada)

Fecha: 8 de octubre de 2026. **El usuario confirmó que la métrica oficial es
accuracy** (la clase neutral es algo menos frecuente). Todo el trabajo previo usaba
F1 macro por instrucción anterior; esta ronda re-optimiza para accuracy.

## Confirmación de la métrica

La OOF en 137 del modelo 009 (L1 C=1) da accuracy **0.89500** y F1 macro 0.89909.
El público de 009 es **0.89555**: coincide con la accuracy local, lo que confirma
independientemente que el score público es accuracy.

## Cambio

Mismo espacio de características (TextFeatures007b `thirds=True`) y L1, pero
**C=2.0** en lugar de C=1.0. Bajo accuracy, una regularización algo menor favorece
la clase neutral (más frecuente en aciertos) y sube la accuracy.

## Evidencia (criterio pre-registrado, `plan_013.json`)

Semillas nuevas 8803/9901/11003 + puerta 137; métrica primaria accuracy:

| Candidata | Δ accuracy medio | Folds ganados /9 | Δ brecha | 137 |
|---|---:|---:|---:|---:|
| l1_c15 (C=1.5) | +0.00333 | 8 | +0.0166 | 0.89667 |
| **l1_c2 (C=2.0)** | **+0.00340** | **8** | +0.0282 | **0.89771** |
| l1_c3 (C=3.0) | +0.00170 | inconsistente | +0.0391 | — |
| L1 C=1 balanced | −0.002 (137) | — | — | 0.89302 |

- **l1_c2 gana** (mayor Δ accuracy): +0.0034 de media, 8/9 folds, puerta 137
  superada (0.89771 vs 0.89500).
- `class_weight='balanced'` empeora la accuracy (0.89302): confirmado que el peso
  balanceado perjudica cuando la métrica es accuracy.
- Un descuento del canal neutral (d≈0.2) añade ~+0.002 sobre 137; es plano y no se
  incorpora para no apilar dos ajustes adaptativos.

## Compromiso con el sobreajuste

C=2 sube la brecha entrenamiento-validación de ~0.06 a ~0.089: memoriza más. La
accuracy OOF, sin embargo, mejora de forma consistente (8/9 folds, 3 semillas nuevas
+ 137). Es un cambio simple de hiperparámetro (no un ensemble), pero la ganancia
pública no está garantizada.

## Verificaciones

- `submission_010.csv`: 3000 filas, `id,answer`, IDs en orden de sample, etiquetas
  válidas, sin vacíos, recarga del Pipeline reproducida. Cambia 112 de 3000
  predicciones respecto a 009.
- `models/modelo_010.joblib` (LinearSVC L1 C=2, n_iter 1833; requiere
  `src/features_007b.py`). SHA-256 en `reports/registro_013.json`. Artefactos
  001–009 intactos.

## Estado

- Público confirmado: 009 = 0.89555; 010 = pendiente (el usuario decide el envío).
- Predicción local de accuracy para 010: ~0.898–0.902 OOF según la semilla.
- No se garantiza 0.90: la ganancia es pequeña (+0.003) y la brecha sube. La
  distancia restante a 0.90 en la escala pública es ~0.004.
