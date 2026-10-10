# Ronda 008 — ponderación del bloque de tercios: sin mejora, conservar 007

Fecha: 8 de octubre de 2026. Métrica exclusiva: F1 macro (`zero_division=0`).
Esta ronda implementa el subconjunto rápido de `revision_007_ideas_008.md`
(referencia y las dos ponderaciones, más el control técnico del split).

## Qué se probó

Candidatas (criterio fijado antes de ejecutar en `plan_008.json`):

| Nombre | Descripción |
|---|---|
| ref007 | 007 exacta: un solo TF-IDF con tercios dentro (reproducción). |
| split_w1 | bloques base/thirds en vectorizadores separados, peso thirds 1.0 (control del split). |
| split_w05 | peso thirds 0.5. |
| split_w15 | peso thirds 1.5. |

El split recombina `[base, w·thirds]` con renormalización L2 de toda la fila; el
vocabulario e IDF de cada bloque se aprenden dentro del fold.

## Resultados (3 semillas nuevas 711/1237/2003, 3 folds cada una)

| Candidata | Δ F1 OOF medio | Folds ganados /9 | Δ brecha media | Veredicto |
|---|---:|---:|---:|---|
| split_w1 | −0.00469 | 1 | +0.00419 | peor |
| split_w05 | +0.00075 | 5 | −0.00073 | neutro |
| split_w15 | −0.01328 | 0 | +0.01134 | claramente peor |

Ninguna candidata cumple las puertas pre-registradas (≥7/9 folds, Δ≥0.003, brecha
no peor, y no perder en 137). Decisión: **conservar 007**; no se generó
`submission_008.csv`.

## Interpretación (hallazgo relevante)

- `split_w1`, el **control técnico**, es peor que `ref007` en las tres semillas
  (media −0.0047). Separar los tercios en su propio vectorizador (IDF y
  normalización independientes) **destruye la ganancia de la 007**.
- `split_w15` confirma que dar más peso a los tercios perjudica (−0.013).
- `split_w05` es neutro, dentro del ruido entre folds.

Conclusión: la mejora de la 007 no proviene del valor informativo aislado de las
etiquetas de tercio, sino del **cambio conjunto de normalización L2 e IDF que
produce añadir esos rasgos al MISMO TF-IDF**. Lo anticipaba
`revision_007_ideas_008.md` ("añadir tercios también cambia la normalización L2").
Por tanto, re-ponderar o aislar el bloque de tercios no es una palanca útil; la 007
es la configuración estable de esta familia.

## Sobre las dos variantes restantes del documento

- **Posición continua suave (triangular)**: implicaría cambiar la transformación TF
  logarítmica por pesos fraccionarios y probarse por separado; el resultado del
  split_w1 sugiere que la ganancia es de normalización, no de la granularidad de la
  posición, por lo que su valor esperado es bajo.
- **Identidad del conector como rasgo**: el propio documento la marca como la de
  mayor riesgo de especificidad/memoria; los antecedentes (global13, edge_bigrams)
  muestran que añadir especificidad posicional/gramatical empeora en este corpus.

Ambas se dejan como rondas futuras a pre-registrar, **no se introducen ahora** tras
ver estos resultados (como exige el documento).

## Estado

- Mejor público confirmado: **007 = 0.88888** (captura registrada en
  `resultado_publico_007.json`); 005 = 0.88111.
- `submission_007.csv` y `models/modelo_007.joblib` siguen siendo los artefactos
  recomendados; los 001–007 permanecen intactos.
- Distancia a 0.90 en la escala pública: 0.01112; la métrica oficial no está
  verificada y no se promete alcanzarla.
- Artefactos de esta ronda: `src/features_008.py`, `src/experimento_008.py`,
  `reports/plan_008.json`, `reports/resultados_008.csv`, `reports/criterios_008.csv`,
  `reports/registro_008.json`, `reports/ejecucion_008.log`.
