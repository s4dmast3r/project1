# Revisión de 009 y propuesta para 010

## Resultado público y verificación

La captura aportada por el usuario muestra submission_009.csv con Public Score **0,89555**. Frente al público confirmado de 007 (0,88888), aumenta **0,00667**. Faltan **0,00445** para 0,90. No hay resultado público verificado de 008 en esta revisión. La captura no identifica la fórmula de la métrica oficial; no convertir la diferencia en un número de aciertos.

Se comprobaron todos los hashes de fuentes y artefactos del registro009. El modelo recargado reproduce el CSV y su orden coincide con sample_submission. Cambia 214 predicciones respecto a007 y 232 respecto a008. La partición archivada mantiene los IDs de train en orden.

## Cambio real

La009 deriva de007, no del NB-SVM008: conserva TextFeatures007b(thirds=True) y cambia LinearSVC L2 C=2 por LinearSVC L1 C=1, dual=False. L1 genera coeficientes cero; el parámetro C también cambió, por lo que no se ha aislado cuánto aporta cada modificación.

Modelo final: 49215 características, 147645 coeficientes entre tres clases, 3252 coeficientes no nulos (2,20%) y 2371 características con algún coeficiente activo (4,82%). Coeficientes y características no son la misma cantidad.

La explicación más consistente con los resultados es que la representación tenía muchos rasgos poco útiles y la regularización redujo su influencia. Esto es una interpretación, no una demostración causal de qué errores específicos del público se corrigieron; no conocemos sus etiquetas.

## Evidencia local comparable

Promedios sobre semillas1009/2027/3053:

| Medida | 007 | 009 |
|---|---:|---:|
| F1 macro OOF | 0,876770 | 0,900647 |
| Brecha entrenamiento-validación | 0,122891 | 0,057281 |

009 ganó9/9 folds. En137: 0,899091 para009 frente a0,877822 para007 (consultar registro para cifra completa). Son métricas de selección sobre datos explorados, no una prueba independiente. Las medias de008 y009 usaron semillas distintas y no deben presentarse como comparación emparejada entre esos modelos.

En3053, F1 por clase de009: negativo0,860563, positivo0,859649, neutral0,985838. Hay463 negativos predichos positivos y434 positivos predichos negativos. Estas897 confusiones explican897/979 errores OOF de esta partición (91,62%); orientan el siguiente trabajo hacia polaridad, conservando la clase neutral. No representan errores del conjunto público.

## Hallazgo en la candidata descartada

En src/features_009.py, NBSVM.fit configura TfidfVectorizer con ngram_range=(1,2), pero CountVectorizer(vocabulary=..., binary=True) conserva su valor por defecto(1,1). Así nunca cuenta los bigramas: sus razones NB quedan log(alpha/alpha)=0 y se anulan al ponderar TF-IDF.

Comprobación sobre300 textos reales de desarrollo:1143 columnas de bigramas; contador actual0 entradas no nulas; contador con ngram_range=(1,2):9683. No se entrenó una candidata nueva ni se tocó la009 para esta comprobación.

El nombre NB-SVM tampoco describe exactamente esa candidata: usa regresión logística y razones sin normalización por masa de clase, mientras008b usa SVM sobre presencia binaria y razones normalizadas. El resultado bajo de nbsvm_word no prueba que NB-SVM o los bigramas sean inútiles. Este defecto NO afecta al Pipeline l1_c1 enviado.

## Siguiente ronda recomendada, acotada

1. **Prioridad: ajustar regularización L1 cerca de009.** Comparar C=0,5;1,0(referencia);1,5;2,0. Fijar esta lista antes de ejecutar, mantener idénticos datos, features y folds. Incluir L2 C=1 como control para separar el efecto de C del de L1. No hacer una búsqueda extensa para alcanzar una cifra deseada.
2. **Diagnóstico de polaridad con predicciones fuera de fold.** Guardar ID, fold, etiqueta, predicción y márgenes de009; revisar una muestra reproducible de confusiones positivo/negativo exclusivamente de desarrollo. Distinguir negación, contraste y distribución de opiniones antes de inventar más rasgos. Toda regla posterior a esta inspección será una hipótesis adaptativa, no una mejora confirmada.
3. **Si el ajuste de C se estanca, estudiar complementariedad con008b.** Evaluar008b y009 en exactamente los mismos folds antes de proponer combinación. Las232 diferencias en eval no dicen cuál acierta. Solo probar una combinación simple predefinida si los errores OOF son complementarios; no promediar márgenes crudos de modelos con escalas diferentes ni optimizar pesos contra el público. No es la primera prioridad.
4. **Corregir la candidata NB de009 antes de volver a juzgarla**, en una versión nueva: alinear ngram_range del contador y vectorizador. Comparar contra008b correcto. El bug es secundario frente al buen resultado de L1 y no justifica rehacer la009 publicada.

## Validación y criterios

Usar009 como nueva referencia. Para elegir C, preferir selección en folds internos y comparación del procedimiento con009 en folds externos agrupados; mantener vocabulario, IDF y cualquier selección dentro de cada entrenamiento. La validación anidada reduce sesgo de la selección de hiperparámetros de esa ronda, pero no borra el historial de exploración del dataset.

Fijar presupuesto y criterio antes de correr, evaluar solo F1 macro, comparar diferencias emparejadas, reportar resultados por clase y brecha. No tratar9folds como9observaciones independientes ni optimizar para la captura pública. No reducir el umbral de aceptación tras ver resultados. Conservar009 si no hay mejora consistente. No usar eval para ajustar, ni añadir etiquetas externas, ni tocar el antiguo test para decidir.

Esta revisión no entrena ni genera submission010. No se promete0,90: la mejora pública de009 es compatible con mejor generalización, pero todavía existe brecha y selección acumulada.

## Fuentes

- [LinearSVC, documentación oficial](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC): L1, coeficientes dispersos y C.
- [Validación anidada y sesgo de selección](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
- Evidencia local: src/experimento_009.py, src/features_009.py, reports/registro_009.json y reports/resultados_009.csv.
