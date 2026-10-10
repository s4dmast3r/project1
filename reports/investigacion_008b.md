# Investigación web y ejecución 008b

Se consultaron fuentes primarias el 8 de octubre de2026. Se usan como métodos y documentación, no como datos de entrenamiento. No se descargaron corpus externos ni se consultó Kaggle. La ronda008 existente se preserva; esta es una continuación diferenciada llamada008b.

## Fuentes y aplicación

1. [Wang y Manning, ACL2012: Baselines and Bigrams](https://aclanthology.org/P12-2018/) y [artículo](https://aclanthology.org/P12-2018.pdf). Propone ponderar rasgos binarios con log-razones de frecuencias por clase antes de ajustar SVM. Se implementa una adaptación multiclase uno-contra-resto con LinearSVC de scikit-learn, suavizado1 y C=1. No se reproduce la interpolación opcional del artículo ni sus resultados. La representación es la propia del proyecto007; se usa presencia binaria, no TF-IDF, en esta candidata. Las razones se aprenden exclusivamente con etiquetas del entrenamiento de cada fold. No garantiza utilidad para este corpus.
2. [scikit-learn: selección de características](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html). Se prueba retener el50% de características según chi-cuadrado, dentro del Pipeline, y renormalizar después. Los rasgos TF-IDF son no negativos. Es una reducción supervisada de capacidad que podría ayudar o descartar información útil; no se presume mejora.
3. [scikit-learn: Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html). Permite normalización por fila manteniendo matrices dispersas. Se usa tras modificar pesos para mantener una escala comparable del vector.
4. [scikit-learn: TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting). El IDF depende de número de documentos y frecuencia documental del término. Separar bloques sobre los mismos documentos no cambia automáticamente el IDF de una característica común, aunque cambien selección de vocabulario y normalización. La inferencia del informe008 de que la ganancia de007 es solo de normalización/IDF no queda demostrada.
5. [scikit-learn: errores comunes y fuga](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage). La selección supervisada y cualquier estadística aprendida se deben ajustar solo dentro del entrenamiento. Se conserva el Pipeline completo por fold, incluida chi-cuadrado y razones NB.

## Implementaciones nuevas

- Referencia007 exacta.
- Ponderación0,5 del bloque third después de transformar con el MISMO TF-IDF007, seguida de normalización conjunta. No se separan vectorizadores.
- Ponderación1,5 bajo el mismo procedimiento.
- Identidad del conector: añadir unigramas combinados con el conector que introduce su cláusula; conservar todos los rasgos007. No se utiliza el conector como decisión de clase. Es una hipótesis propia, no una técnica cuya mejora garantice un artículo consultado.
- Chi-cuadrado50% y LinearSVC C=2.
- NB-SVM multiclase adaptado: mismo analizador007, conteos binarios, razones por clase vs resto, SVM C=1 por clase, decisión por mayor margen.

El control de ponderación1 fue contrastado numéricamente con007 sobre300 textos reales de desarrollo: mismas columnas y valores dentro de tolerancia1e-12. También se comprobó igualdad de IDF por rasgo común entre vectorización conjunta y separada. Ver pruebas_008b.json. No se generaron reseñas sintéticas.

## Protocolo acotado

Seis configuraciones x tres folds x semillas711/1237/2003:54 ajustes de screening. Se reutilizan las particiones de008 deliberadamente para comparar; no se llaman prueba nueva. Hasta18 ajustes históricos en137 solo si hay candidatos aptos. Métrica F1 macro con etiquetas explícitas y zero_division=0.

Puertas fijadas antes de ejecutar: ganar al menos7/9 folds, delta medio de F1 OOF al menos0,003, brecha media no mayor que007; luego no perder en137. Si varias aprueban, mayor F1 medio; empate menor brecha. No se alterarán después de ver resultados. Todos esos scores son de selección, no evaluación externa independiente.

No se prueba una búsqueda creciente hasta alcanzar0,90. Si nada cumple, se conserva007 y se documenta el resultado negativo. Ningún éxito en otra publicación promete el mismo rendimiento aquí. La prueba local antigua permanece excluida; eval se usa únicamente si procede generar un CSV final. Modelos y documentos anteriores permanecen intactos.

## Qué no se ejecuta

No se añade posición triangular suave en esta ronda: requeriría definir y aislar TF para conteos fraccionarios. Tampoco nuevos ensembles, redes neuronales, embeddings, cambios de etiquetas, datos externos, stopwords externas ni augmentación. El presupuesto prioriza alternativas clásicas distintas de las ya ejecutadas y una corrección del control de ponderación.
