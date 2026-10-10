# Versión 007 — confirmación pre-registrada de `thirds` sobre 005

Fecha: 8 de octubre de 2026. Métrica exclusiva: F1 macro (`zero_division=0`).

## Qué es la 007

Es la representación 005 sin cambios más **una etiqueta de tercio del texto
completo** para cada palabra (`third0/third1/third2`), añadida al canal global:

- Canal global: unigramas y bigramas (como 005).
- Canal posicional: unigramas por cláusula con `start0-3`/`end0-3` (como 005).
- Cierre discursivo con interacción `epilogueTrue/False` (como 005).
- **Nuevo**: `third{0,1,2}:palabra` según si la palabra está en el primer, segundo
  o tercer tercio de la reseña completa.
- Sin canal de caracteres; TF-IDF `min_df=3`, `max_features=160000`,
  `sublinear_tf=True`; `LinearSVC(C=2, dual='auto', max_iter=10000, random_state=42)`.

Implementación: `src/features_007b.py` (los valores por defecto reproducen
exactamente 005; verificados token a token y por vocabulario) y
`src/confirmar_007b.py`.

## Evidencia (todo medido antes de decidir)

Ronda de exploración 007b (semillas 271/599/913, 3 folds cada una):

| Candidata | Efecto vs ref005 |
|---|---|
| global13 (trigramas globales) | −0.007 a −0.009 OOF: descartada |
| mid_bucket (palabras centrales) | −0.002 a −0.003: descartada |
| edge_bigrams (bigramas de borde) | −0.006 a −0.007: descartada |
| **thirds** | **+0.0021 a +0.0041 OOF, brecha menor en las 3 semillas, 9/9 folds ganados** |

`thirds` no alcanzó el umbral de +0.005 por semilla fijado para la ronda 007b, por
lo que esa ronda concluyó "conservar 005" (registro_007b.json). Para no relajar un
criterio después de ver resultados, se fijó **antes** un criterio de confirmación
sobre semillas nunca usadas (1142, 2087, 3313) y la puerta histórica 137:

| Puerta | Criterio | Resultado |
|---|---|---:|
| Gana folds emparejados | ≥7 de 9 | **9/9** ✓ |
| Delta medio F1 OOF | ≥0.002 | **+0.00328** ✓ |
| Brecha media | ≤ ref005 | **−0.00324** ✓ |
| Histórica 137 | no perder | **0.87782 vs 0.87338** ✓ |

En total `thirds` gana 18/18 folds emparejados (9 en exploración + 9 en
confirmación), con media +0.0028 a +0.0033 y brecha siempre menor. Aprobada por la
regla pre-registrada: se generó `submission_007.csv`.

## Métricas y verificaciones

- OOF 007 (semilla 137): 0.877822 frente a 0.873382 de 005 (+0.0044).
- Per class (semilla 3313): negativo 0.8246, neutral 0.9882, positivo 0.8215.
- `submission_007.csv`: 3000 filas, `id,answer`, IDs en orden de sample, etiquetas
  válidas, sin vacíos, lectura ida y vuelta y recarga del Pipeline reproducida.
  Cambia 68 de 3000 predicciones respecto a 005 (cambio conservador).
- Modelo: `models/modelo_007.joblib` (requiere `src/features_007b.py`). SHA-256 en
  `reports/registro_confirmacion_007b.json`; artefactos 001–006 intactos.

## Límites y riesgo

- La mejora local es pequeña (+0.003 media) aunque muy consistente (18/18 folds) y
  reduce la brecha: es una simplificación estructural leve, no un añadido complejo
  como el stacking de la 006.
- El desarrollo y sus particiones se han explorado en muchas rondas; los resultados
  son comparación emparejada, no una prueba virgen. Cambiar semillas no crea datos
  nuevos.
- No se puede garantizar que el público suba: el público 006 cayó pese a una mejora
  local similar. La decisión de enviar es del usuario.
- La meta 0.90 no se alcanza en ninguna evaluación local honesta (mejor OOF 0.878).

## Cómo se llegó (historial completo)

- 003 (posicional 1–3 g + caracteres) → 004 (1–2 g) → 005 (1 g, sin caracteres):
  cada simplificación mejoró (público 0.84333 → 0.86111 → 0.88111).
- 006 (stacking SVC/LR/NB): mejora local +0.005 que no transfirió; público cayó a
  ~0.87. Lección: no seleccionar sobre folds quemados ni con márgenes bajo el ruido.
- 007 (negación, contraste, C=1/min_df=5): ninguna candidata pasó (±0.001).
- 007b (estructura del espacio de características): trigramas globales, bucket
  medio y bigramas de borde empeoran; **tercios del texto completo** mejora de forma
  consistente y reduce brecha.
- Confirmación pre-registrada en semillas nuevas: aprobada; se produce la 007.
