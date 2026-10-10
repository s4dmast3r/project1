# Investigación y protocolo 011b

Fecha: 9 de octubre de 2026. El usuario confirmó en este chat **accuracy principal, F1 macro como control**. La captura identifica submission_010.csv y Public Score 0,89777. No se consultó Kaggle ni se descargaron datos externos.

## Auditoría de 010

La entrega010 real viene de src/experimento_013.py, no del primer experimento llamado010. Se verificaron hashes de fuentes/modelo/CSV contra registro_013.json y el modelo recargado reproduce el CSV. LinearSVC L1 C=2 conserva las características007 y usa5491 coeficientes no nulos, frente a3252 de009(C=1).

En las mismas semillas8803/9901/11003, accuracy media009=0,895660 y010=0,899062; F1 macro009=0,899694 y010=0,903358; brecha de accuracy0,060799→0,088976. El público subió0,89555→0,89777 (+0,00222). Hay una mejora observada con menor regularización, acompañada de más ajuste al entrenamiento. No se puede atribuir la subida pública a una clase concreta sin etiquetas públicas.

Correcciones a informes anteriores: que accuracy local sea cercana al score público no confirma la fórmula oficial; aquí se usa por confirmación explícita del usuario. Los ensayos negativos no demuestran un techo absoluto de ML clásico ni que C=1 sea un óptimo universal. La ronda011 anterior se conserva como resultado negativo; esta continuación se llama011b y solo exportará submission_011 si cumple los criterios.

## Fuentes primarias consultadas

- [TfidfVectorizer, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html): parámetros binary, use_idf, sublinear_tf y normalización. Permiten aislar si presencia o frecuencia y ponderación documental ayudan. Se comparan tres variantes; no se presupone que IDF sea siempre beneficioso para sentimiento.
- [LogisticRegression, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): regresión logística con regularización L1. Se usa liblinear dentro de OneVsRestClassifier, C=4, misma representación010. Cambiar pérdida y C es una alternativa de pipeline, no una ablación causal aislada. La regresión multinomial y elastic net son alternativas documentadas, pero no se añaden para limitar presupuesto.
- [Exploiting Discourse Relations for Sentiment Analysis, Wang et al., COLING2012](https://aclanthology.org/C12-2128/): motivación para estudiar relaciones discursivas y conectores. Nuestra característica palabra×conector es una aproximación propia; no reproduce el sistema ni transfiere resultados del artículo. Se reutiliza ConnectorAnalyzer de008b con L1; aquella prueba había usado L2.

Las fuentes orientan métodos. No se incorporan corpus, lexicones externos, embeddings, redes ni ejemplos sintéticos.

## Configuraciones fijadas antes de observar

Todas conservan las características de discurso/posición007 y min_df=3, max_features=160000. Normalización L2 por documento.

1. ref010: TF sublineal×IDF + LinearSVC L1 C=2.
2. binary_idf: presencia binaria×IDF + mismo SVM.
3. binary_noidf: presencia binaria sin IDF + mismo SVM.
4. tf_noidf: TF sublineal sin IDF + mismo SVM.
5. connector_l1: representación010 más identidad de conector×palabra, TF sublineal×IDF + mismo SVM.
6. logistic_l1: representación010 + logística uno-contra-resto L1 C=4.

Se comprobaron las seis interfaces sobre300 textos reales de desarrollo sin medir scores de selección. Convergencia obligatoria: ConvergenceWarning tratado como error y límite de iteraciones verificado.

## Selección y protección frente a fuga

Semillas8803/9901/11003 reutilizadas deliberadamente: tres folds agrupados por semilla, idénticos para todas las candidatas. Solo9600 filas de desarrollo participan en selección. Vocabulario/IDF/modelo se ajustan exclusivamente en entrenamiento de cada fold. El test local antiguo queda excluido de selección; ya no es una prueba virgen del proyecto. Eval solo entra después de cerrar selección para exportar.

54 ajustes de pipeline y hasta18 adicionales condicionales en137. Reemplazo exige: al menos7/9 victorias en accuracy; delta medio de accuracy>=0,003; F1 macro medio no menor; brecha media de accuracy no mayor; no perder accuracy ni F1 en137. Si varias pasan, mayor accuracy media; empate menor brecha. No se modifican puertas ni se amplía la búsqueda tras ver resultados.

Esto es CV de selección, no una evaluación independiente. Reutilizar datos, incluso cambiando semillas, permite sobreajuste de selección acumulado; folds repetidos no son independientes. Brecha menor no garantiza ausencia de sobreajuste. No se promete superar0,90 público. Un resultado negativo se documentará sin fabricar un CSV equivalente a010.
