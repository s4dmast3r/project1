# Auditoría011 y estrategias012b

Fecha: 9 de octubre de2026. Accuracy principal, F1 macro control, según confirmación previa del usuario. Captura011: score0,89888; campo privado sin valor visible. No se accedió a Kaggle.

## Qué cambió en011

El archivo real proviene de src/producir_011.py y registro_produccion_011.json. Se verificaron hashes de modelo y CSV. Los coeficientes de011 son idénticos a010: mismo LinearSVC L1 C=2 y TextFeatures007b(thirds=True). Se suma0,2 a los márgenes negativo/positivo antes del argmax, equivalente a restar0,2 a neutral. Se verificó la reproducción del CSV con esa regla. Cambia4 de3000 predicciones; el score visible sube0,89777→0,89888 (+0,00111). No sabemos qué filas se evalúan públicamente ni sus etiquetas.

Correcciones documentales: el script sí reentrena el modelo, aunque el informe dice que no; produce los mismos coeficientes. El .joblib guarda la base sin corrección, por lo que predict() difiere del CSV en4filas; la corrección solo vive en el script. La referencia nueva encapsula la regla completa por nombres de clase, sin modificar los artefactos previos.

La menor frecuencia de neutral no prueba por sí sola que deba restarse0,2 a un margen SVM. Se trata de un ajuste empírico de decisión, no una corrección probabilística de prior demostrada. Los informes afirman validación anidada sobre12007/13001/14009, pero esta revisión no encontró sus tablas numéricas ni el código que la reproduce en los archivos examinados. Se distingue esa afirmación de la evidencia verificable de exportación y de la nueva evaluación reproducible.

Tampoco las pruebas negativas anteriores demuestran un techo absoluto de0,899 para ML clásico, ni justifican reetiquetar ejemplos. Se mantienen las restricciones: sin datos/lexicones externos, redes, embeddings ni reseñas sintéticas; no se cambian etiquetas.

## Investigación y nuevas hipótesis

- [OneVsOneClassifier, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html): aprende clasificadores para pares de clases y combina sus decisiones. Permite estudiar si el esquema uno-contra-resto actual perjudica la separación positivo/negativo. Se prueba L1 C=2, mismas características. No se asume mejora.
- [CalibratedClassifierCV, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html): con ensemble=False usa predicciones fuera de fold para aprender la calibración y ajusta la base final con el entrenamiento disponible. Se prueba sigmoid, con tres folds internos agrupados. El Pipeline entero, incluyendo vocabulario/IDF, está dentro del calibrador. Mejorar calibración no garantiza mejorar accuracy.
- Especialista de polaridad: hipótesis propia motivada por el diagnóstico previo de confusión positivo/negativo. Mantiene exactamente la decisión neutral de011; para textos que011 considera no neutrales, usa un SVM binario entrenado solo con ejemplos positivos/negativos del entrenamiento. Se comparan C=1 yC=2. No se interpreta como regla de etiqueta ni se ajusta con eval.

## Configuraciones

1. ref011: modelo010 con descuento neutral0,2 encapsulado en predict.
2. polarity_c1: misma puerta neutral; especialista positivo/negativo L1 C=1.
3. polarity_c2: especialista L1 C=2.
4. ovo_l1: tres pares de clases con L1 C=2 sobre representación007.
5. sigmoid: calibración sigmoide por clase de la base010, con OOF interna y refit. No se añade el descuento manual encima de las probabilidades.

Se verificaron interfaces sobre300 filas reales de desarrollo sin calcular scores para seleccionar candidatos. CheckedL1 comprueba convergencia en cada ajuste, también en los folds internos. Las decisiones de011 se reproducen usando nombres de clase, sin asumir posiciones fijas.

## Presupuesto y puertas fijadas antes de resultados

Semillas8803/9901/11003 reutilizadas; tres folds agrupados cada una. Solo9600 filas de desarrollo.45 ajustes externos de pipeline; cada especialista/calibrador puede entrenar varias bases. Hasta15 ajustes externos históricos condicionales en137 y un ajuste final si hay ganadora. No se amplía búsqueda ni cambia umbral según resultados.

Umbral nuevo declarado antes de medir: delta accuracy medio>=0,001; victorias>=7/9; F1 macro medio no inferior; aumento de brecha accuracy media<=0,005. Luego no perder accuracy ni F1 en137. Si varias aprueban: mayor accuracy media; empate menor brecha. Este umbral es distinto del0,003 de011b; se busca detectar mejoras pequeñas con una ronda separada y no se reclasifican los resultados anteriores. No implica significación estadística. La tolerancia de brecha acepta un aumento pequeño y explícito, no garantiza menos sobreajuste.

El test local histórico queda excluido de selección; eval solo se utiliza para auditoría de artefactos previos y exportación posterior a selección. Las calibraciones se aprenden solo con predicciones internas del entrenamiento externo. Para el especialista, el vectorizador aprende de todo el entrenamiento externo y el SVM binario únicamente del subconjunto polar; ninguna etiqueta o texto de validación entra en ese ajuste.

Las particiones son de selección, no prueba externa independiente. La calibración interna evita fuga dentro de cada ajuste pero no borra selección adaptativa acumulada entre experimentos. No se atribuye independencia a nueve folds repetidos. Si no hay ganadora, conservar011 y documentar el resultado; si la hay, exportar012 con predict completo reproducible. No se promete superar0,90 público.
