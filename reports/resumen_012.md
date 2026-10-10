# Diagnóstico de polaridad y ronda 012 — la 009 es el techo

Fecha: 8 de octubre de 2026. Métrica local exclusiva: F1 macro (`zero_division=0`).

## Diagnóstico de polaridad (errores positivo↔negativo de la 009)

Sobre la OOF de 137 (F1 macro 0.899091), los errores pos↔neg (439 pos→neg y
485 neg→pos) comparados con los aciertos pos/neg (5737):

| Señal | Aciertos pos/neg | Errores pos↔neg |
|---|---:|---:|
| Con negación (`no/nunca/sin/ni/…`) | 48,8 % | 67–73 % |
| Con contraste (`pero/aunque/sin embargo/…`) | 51,6 % | 61–64 % |
| Marcas de opinión negativa (media) | 0,75 | 1,30–1,41 |
| Marcas de opinión positiva (media) | 0,21 | 0,35–0,46 |

Confirmado: los errores son **reseñas de aspecto mixto** con señal positiva y
negativa a la vez, unidas por conectores de contraste o negación. Es exactamente la
dificultad que el proyecto señala. No hay una regla de posición simple: la etiqueta
depende de ponderar cláusulas en conflicto.

## Ronda 012 — rasgos densos de sentimiento (motivados por el diagnóstico)

Se añadieron a la 009 conteos densos explícitos: negación, contraste, número de
cláusulas y, en una variante, palabras de polaridad de un léxico interno aprendido
dentro del fold (razón NB). Criterio pre-registrado, semillas nuevas 5501/6601/7703.

| Candidata | Δ F1 OOF medio | Folds ganados | Veredicto |
|---|---:|---:|---|
| dense_markers (sin léxico) | ~0.000 | 1–2/3 | neutro (redundante) |
| dense_lexicon (léxico interno) | −0.0058 a −0.0113 | 0/3 | peor |

Resultado: **ninguna candidata supera las puertas; se conserva 009 y no se genera
`submission_010.csv`.** Los conteos de marcadores son redundantes (el modelo ya ve
«no», «pero» como rasgos dispersos) y el léxico interno añade ruido que aumenta la
brecha.

## Estado consolidado de estrategias probadas

| Estrategia | Resultado frente a 009 |
|---|---|
| C en L1 (0.5/1.5/2.0) y control L2 C=1 | C=1 óptimo; L2 retrocede (aisla L1) |
| Trigramas globales, bigramas de borde, char_wb, min_df=2 bajo L1 | neutros (±0.001) |
| NB-SVM (008b) y voto/stacking con 009 | peor (errores 68 % solapados) |
| Conteos densos de sentimiento / léxico interno | neutro o peor |
| Calibración de umbral por clase | +0.0016 (ruido) |

## Conclusión honesta

La 009 (LinearSVC L1 C=1 sobre las características 007) corrigió un sobreajuste
extremo y agotó ese margen: F1 macro OOF ~0.90, público 0.89555. **No hay un
`submission_010` que mejore a la 009** dentro de las restricciones de la etapa
(ML tradicional + NLP clásico, sin datos/lexicones externos, sin transformadores).
Generar un envío 010 sería una variación de ruido que arriesga repetir la caída
pública de la 006.

**Punto pendiente con el usuario (crítico):** la métrica oficial nunca se verificó
(las capturas no muestran su nombre). Todo el trabajo local usa F1 macro por
instrucción previa, pero si la competencia evalúa con otra fórmula (p. ej. accuracy o
F1 ponderado), el objetivo cambia. Compartir el texto de las reglas permitiría
confirmar o reorientar la optimización; no se accede a Kaggle por iniciativa propia.
