# Versión 006: recomendaciones contrastadas

Candidato seleccionado: **stack_svc_lr_nb**.

| F1 macro | Referencia 005 | Procedimiento 006 |
|---|---:|---:|
| Selección CV | 0.874163 | 0.878010 |
| Brecha entrenamiento-CV | 0.125293 | 0.121140 |
| Predicciones OOF externas agrupadas | 0.873382 | 0.878421 |

F1 entrenamiento de selección: 0.999150. Media de folds externos: 0.878413 ± 0.004300. El F1 sobre todas las predicciones externas no es la misma operación que promediar F1 por fold.

## Cambios y evaluación

Se contrastaron referencia 005, C=1/min_df=5, eliminación de interacciones de epílogo y stacking SVC/LR/NB. Solo F1 macro dirige la selección, con regla previa de control de brecha y mejora mínima. Los detalles se fijaron en reports/plan_006.json y src/modelos_006.py.

La CV externa repite selección interna. Cada stacking genera sus propios scores OOF de bases dentro del entrenamiento recibido; no reutiliza scores globales para entrenar el agregador. QuantileTransformer aprende únicamente sobre OOF de entrenamiento, nunca calcula rangos a partir del lote eval. Las bases no necesitan ser estadísticamente independientes: su utilidad debe comprobarse fuera de entrenamiento.

La cifra exploratoria 0,8805 del documento del auditor no se usa como resultado verificado: no se encontró su implementación y artefactos intermedios en las carpetas revisadas. Esta implementación tampoco pretende reproducir exactamente un ranking por fold no especificado.

El público 005 aportado por el usuario es 0,88111; no se consultó la plataforma. El público 006 es desconocido. Meta local 0,90: no alcanzada en esta OOF. No se extrapola del score público anterior.

## Sobreajuste y límites

La brecha reportada es diagnóstica; su reducción no demuestra ausencia de sobreajuste. El mismo desarrollo se ha explorado en varias versiones. La CV anidada protege los ajustes de esta ronda, pero no borra adaptación histórica ni dependencias entre plantillas. La prueba local antigua no se reutilizó. Tras selección, el modelo final se ajusta con todo train; si gana stacking, también se regeneran scores OOF de todo train para su meta-modelo. eval solo se utiliza para predecir.

Los scores externos no se utilizaron para cambiar parámetros de esta ronda. No se ejecutaron stemming, stopwords, árboles ni KNN, para limitar coste y decisiones adaptativas. No se añadieron datos externos o sintéticos.

## Entregables

CSV nuevo creado: True. Usar submissions/submission_006.csv; modelo models/modelo_006.joblib. Conservar src/modelos_006.py, src/features_004.py y src/experimento.py para las dependencias del Pipeline.

Duración medida: 321.6 segundos. Los archivos anteriores permanecen intactos. Registro: reports/registro_006.json. Resultados completos: reports/resultados_cv_006.csv, cv_006_interna_*.csv y cv_006_anidada.csv.
