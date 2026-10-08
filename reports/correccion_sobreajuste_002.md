# Corrección de sobreajuste

La regla se guardó antes de la búsqueda en plan_regularizacion_002.json: Entre configuraciones a <=0.01 de la máxima accuracy CV, elegir la menor brecha media entrenamiento-validación (truncada en cero); desempatar por accuracy CV descendente. Regla fijada antes de ejecutar. No usar prueba local.

Se compararon 25 configuraciones en los mismos cinco folds de desarrollo (9600 filas). TF-IDF permanece dentro del Pipeline. Se probaron vocabularios limitados, min_df=5 y distintos niveles de regularización. El modelo original participa como referencia. No se usa eval para ajustar vocabulario ni seleccionar modelos.

Configuración elegida: `{'clf__C': 4.0, 'tfidf__max_features': 1000, 'tfidf__min_df': 5, 'tfidf__ngram_range': (1, 1)}`.

| Accuracy | Original | Corrección |
|---|---:|---:|
| Entrenamiento (media folds) | 0.982526 | 0.823568 |
| Validación (media folds) | 0.731979 | 0.726042 |
| Brecha | 0.250547 | 0.097526 |

F1 macro CV corregido: 0.737525 ± 0.007455. Accuracy CV: 0.726042 ± 0.008149.

La tolerancia de un punto porcentual limita la pérdida de validación permitida para favorecer una brecha menor. No se minimiza la brecha sin restricciones: un modelo que siempre predice la mayoría puede tener una brecha pequeña y ser inútil. La mejora de la brecha se refiere a estos folds; no demuestra ausencia absoluta de sobreajuste. La selección entre múltiples configuraciones puede introducir optimismo en CV, y reutiliza los folds del experimento anterior.

No se vuelve a evaluar la prueba local original: ya se observó y motivó esta revisión. Su accuracy anterior NO corresponde al modelo corregido. La evaluación independiente del nuevo modelo queda pendiente. No se garantiza un puntaje en Kaggle y no se realizó ningún envío.

Tras cerrar parámetros, se ajustó sobre todo train para producción. El CSV nuevo tiene 3000 filas con IDs en el orden requerido y predicciones verificadas tras recargar el modelo. Los originales se conservan. Archivo recomendado para enviar manualmente: submissions/submission_002_regularizado.csv. El número identifica una versión local, no un envío efectuado.
