# Ronda 008b: estrategias investigadas y ejecutadas

## Resultado

Candidata seleccionada: **nbsvm**. CSV creado: **True**.

| Modelo | F1 macro OOF medio | Diferencia frente a 007 | Brecha entrenamiento-validación | Folds ganados |
|---|---:|---:|---:|---:|
| nbsvm | 0.892841 | +0.015556 | 0.107165 | 9/9 |
| chi2_50 | 0.889295 | +0.012009 | 0.109054 | 9/9 |
| joint_w15 | 0.877792 | +0.000506 | 0.121858 | 5/9 |
| ref007 | 0.877286 | +0.000000 | 0.122364 | — |
| connector_identity | 0.877041 | -0.000245 | 0.122803 | 4/9 |
| joint_w05 | 0.875240 | -0.002045 | 0.124407 | 2/9 |

Los valores son medias de F1 macro calculado sobre predicciones fuera de fold (OOF) en tres particiones de desarrollo. La brecha es la media de F1 de entrenamiento menos F1 de validación por fold. La brecha menor no prueba ausencia de sobreajuste.

## Qué se implementó

- `ref007`: reproducción del modelo 007, TF-IDF con posición por tercios y LinearSVC C=2.
- `joint_w05` y `joint_w15`: pesos 0,5 y 1,5 para tercios en el mismo TF-IDF y normalización final L2. El control de peso 1 reproduce numéricamente 007.
- `connector_identity`: características que unen la identidad del conector con palabras de su cláusula, sin reglas manuales de etiqueta.
- `chi2_50`: selección supervisada del 50 % de características dentro de cada fold, seguida de normalización y LinearSVC C=2.
- `nbsvm`: presencia binaria de las características de discurso de 007, ponderación logarítmica Naive Bayes por clase frente al resto y tres SVM C=1; gana la clase de mayor margen. Es una adaptación multiclase del artículo, sin su interpolación opcional.

Investigación y fuentes: [detalle](investigacion_008b.md), [Wang y Manning, 2012](https://aclanthology.org/P12-2018/), [selección de características](https://scikit-learn.org/stable/modules/feature_selection.html), [prevención de fuga](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage). Los recursos externos aportan métodos, nunca ejemplos de entrenamiento.

## Protocolo y decisión

>=7/9 folds ganados; delta medio OOF >=0.003; brecha media no mayor; no perder en137. No alterar después de ver resultados.

Se usaron las mismas semillas 711, 1237 y 2003 de la ronda 008 para comparar con 007. La ronda anterior se conserva. Las candidatas elegibles se compararon también en la partición histórica 137:

- chi2_50: 0.885014, referencia 007: 0.877822; supera puerta: True.
- nbsvm: 0.888992, referencia 007: 0.877822; supera puerta: True.

Tras cerrar la selección se ajustó la candidata elegida con los 12000 ejemplos etiquetados. Se exportaron `models/modelo_008.joblib` y `submissions/submission_008.csv`. El CSV conserva el orden del ejemplo de envío, con 3000 identificadores únicos, columnas `id,answer` y etiquetas válidas. No se ha enviado a la plataforma.

## Sobreajuste y límites

Solo 9600 ejemplos de desarrollo participan en selección. Los 2400 del antiguo test local no se reutilizan para seleccionar; ya habían sido observados en la historia del proyecto y no se presentan como prueba virgen. Tras decidir el modelo, incluirlos en el ajuste final para predecir eval es legítimo.

Se mantienen grupos disjuntos por texto normalizado. Vocabulario, IDF, chi² y cocientes Naive Bayes se aprenden únicamente con la parte de entrenamiento de cada fold. Las etiquetas de validación solo sirven para calcular F1 macro. Eval no interviene en estos ajustes ni decisiones.

Todas estas métricas son de selección: se han explorado repetidamente los mismos datos. Los folds y semillas no son observaciones independientes; no se atribuye significación estadística a contar victorias. El criterio fijo reduce decisiones oportunistas en esta ronda, pero no elimina el sobreajuste de selección acumulado. Tampoco se garantiza ausencia de duplicados semánticos con una normalización simple.

La captura aportada registra 0,88888 público para 007. No verifica la fórmula oficial de la métrica. El score público de 008 es desconocido y el F1 local no promete 0,90 público. El entrenamiento casi perfecto, incluso con una brecha menor, exige conservar esta cautela.

La conclusión causal del informe anterior 008 era demasiado fuerte: separar bloques altera su normalización, pero no el IDF de características comunes con igual frecuencia documental. El control real de esta ronda comprueba esta igualdad; no se concluye que toda ponderación de tercios sea inútil.

## Reproducibilidad

Tiempo del experimento: 289.4 segundos. Artefactos anteriores intactos: True.

Desde la raíz: `.venv/bin/python -m src.experimento_008b`. El registro existente impide repetir accidentalmente la ronda. Generar este informe y notebook: `.venv/bin/python -m src.informe_008b`.

Los hashes, versiones, resultados por fold, predicciones OOF y criterios están en los archivos `registro_008b.json`, `folds_008b.csv`, `predicciones_oof_008b.csv` y `criterios_008b.csv` de reports. El notebook reproduce el análisis guardado; no vuelve a entrenar.
