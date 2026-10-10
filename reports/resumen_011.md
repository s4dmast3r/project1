# Versión 011 — descuento de neutral sobre 010 (métrica accuracy)

Fecha: 9 de octubre de 2026. Métrica oficial: accuracy (confirmada por el usuario).

## Cambio

Mismo modelo que la 010 (LinearSVC L1 C=2 sobre las características 007), con una
**corrección de prior en la regla de decisión**: se descuenta la clase neutral con
d=0.2, es decir `argmax(s_negativo+0.2, s_neutral, s_positivo+0.2)`. Como neutral es
la clase menos frecuente, el prior de accuracy favorece ligeramente positivo/negativo.

d=0.2 se seleccionó por validación anidada en desarrollo (valores 0.15–0.30, media
~0.2). No se reentrenó nada; solo cambia la decisión sobre los mismos scores.

## Evidencia

- Ganancia de accuracy en semillas nuevas 12007/13001/14009: +0.0006 a +0.0009 de
  media, consistente (3/3 semillas positivas), con d~0.2 estable.
- Cambio en eval: **4 de 3000 predicciones** pasan de neutral a positivo/negativo
  frente a la 010. Es una corrección mínima, no un reajuste masivo.

## Verificaciones

- `submission_011.csv`: 3000 filas, `id,answer`, orden de sample, etiquetas válidas,
  sin vacíos; recarga del modelo con el descuento reproduce las predicciones.
- SHA-256 en `reports/registro_produccion_011.json`; artefactos 001–010 intactos.

## Límites honestos

- La ganancia es **pequeña y dentro del ruido entre folds** (+0.0007 local; 4 filas
  de 3000). No garantiza superar el público 0.89777 de la 010.
- La secuencia pública 005→007→009→010 muestra rendimientos decrecientes
  (+0.0067, +0.0022, y ahora ~+0.001 esperado): el modelo converge a un techo de
  ~0.898–0.899 en accuracy.
- Los 0.0022 restantes hasta 0.90 no se alcanzan con las estrategias probadas
  (regularización, representación, ensembles, calibración, umbrales). Alcanzar 0.90
  de forma fiable requeriría recursos fuera de esta etapa (lexicones/embeddings
  externos, transformadores, o más datos etiquetados), o relabelar reseñas de
  aspecto mixto, todo lo cual está excluido por las restricciones vigentes.
