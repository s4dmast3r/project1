# Versión 005: simplificación de 004 con F1 macro

Configuración seleccionada: `{'name': 'ng1_nochar', 'max_ngram': 1, 'min_df': 3, 'char_weight': 0.0, 'max_features': 160000}`. Clasificador LinearSVC C=2.

| F1 macro | Referencia 004 | Procedimiento 005 |
|---|---:|---:|
| CV usada para selección | 0.851422 | 0.874163 |
| Brecha train-CV | 0.148529 | 0.125293 |
| OOF externa agrupada | 0.850578 | 0.873382 |

F1 de entrenamiento de la selección 005: 0.999456. F1 medio de folds externos: 0.873373 ± 0.004074. La media de folds y F1 de predicciones agrupadas no son la misma operación.

## Interpretación

El auditor mejoró 004 quitando trigramas de cláusulas. Esta ronda contrasta más simplificación, frecuencia mínima y ausencia de caracteres con un presupuesto previo de seis configuraciones. Selección exclusivamente por F1 macro y control de brecha: F1 macro exclusivamente; exigir brecha <= brecha de ref004 +0.002; entre elegibles a <=0.003 del mejor F1, menor brecha; desempate mayor F1. Si no mejora F1 de ref004 en >=0.002, conservar referencia.

La evaluación externa repite la selección dentro del entrenamiento de cada fold: no selecciona directamente por los scores externos como hizo 004. Desarrollo ya fue examinado en rondas anteriores, por lo que esta separación no equivale a una prueba nueva. La referencia 004 también fue elegida históricamente con estos folds; no se oculta esa adaptación.

La meta 0,90 no se alcanza en la estimación externa local. Un F1 mayor no elimina el sobreajuste. La brecha persiste y las familias de textos o plantillas podrían inducir dependencia no capturada por los grupos de minúsculas/espacios.

La antigua prueba local se excluyó. Tras cerrar selección, todo train se usa para producción; eval solo para predecir. No se consultó Kaggle. El público conocido de 004 es 0,86111; no es el resultado de esta versión ni se usó para ajustar parámetros. El público 005 está pendiente.

## Artefactos y verificaciones

CSV nuevo creado: True. Archivo: submissions/submission_005.csv. Modelo: models/modelo_005.joblib. El Pipeline requiere src/features_004.py.

Se verificaron matrices dispersas, grupos disjuntos, convergencia de los nuevos SVM, conservación de datos y artefactos anteriores. Si se generó CSV, se verificaron esquema id,answer, cantidad, IDs únicos, orden de sample_submission, etiquetas, ausencia de vacíos, lectura de ida y vuelta y predicciones tras recarga. Duración total medida del experimento: 644.2 segundos.

La numeración 005 es versión de archivo. Los envíos reales reportados por el usuario son 002, 003 y 004; esta sería la siguiente carga manual, salvo otros envíos no comunicados.
