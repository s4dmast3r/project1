# Revisión de la 007 enviada e hipótesis para 008

## Identidad y verificaciones

La primera ronda007 (negación, contraste y C=1/min_df=5) no produjo un CSV. El envío posterior proviene de la ronda007b y `src/confirmar_007b.py`. El registro operativo es `reports/registro_confirmacion_007b.json`; el registro007 original permanece correcto como historial del intento anterior.

Se revisaron `features_007b.py`, `confirmar_007b.py`, sus planes, reportes y log. Se verificaron las huellas de fuentes, particiones, modelo y CSV contra el registro de confirmación. El modelo recargado reproduce el CSV en el orden requerido, con3000 IDs únicos. Son68 cambios frente a005.

El modelo es TextFeatures007b(thirds=True) + LinearSVC(C=2). Tiene49215 características finales, de las cuales4792 corresponden a tercios. Estas cantidades son del ajuste final con todo train, no de cada fold.

## Cambio real respecto a005

Mantiene normalización sin acentos, unigramas/bigramas globales, unigramas por posición de cláusula start/end, tratamiento del cierre discursivo e interacciones de epílogo. Sin caracteres, sin stacking. TF-IDF conserva min_df=3, max_features=160000, sublinear_tf=True y L2.

Añade `third0:palabra`, `third1:palabra` o `third2:palabra`, según `min(2, 3*i//N)`, donde i es el índice de la palabra y N el número total de palabras. Se calcula antes de excluir cierres del cuerpo: comprende toda la reseña. No etiqueta sentimientos ni aplica una regla de última oración.

La nueva información complementa las posiciones de cláusulas, cuya categoría se satura en3 desde cada extremo. Es una hipótesis razonable que la posición relativa global desambigüe algunos casos; no es una explicación causal demostrada de cada corrección.

Todos los rasgos están dentro del MISMO TF-IDF, no en un canal normalizado separadamente. Por eso añadir tercios también cambia la normalización L2 y puede afectar el límite de vocabulario y las contribuciones relativas. No atribuir toda mejora a “comprensión de la estructura”. Añade capacidad, aunque sea un cambio sencillo; no es una simplificación estricta de005.

## Evidencia

- Comparación posterior en semillas1142/2087/3313: F1 macro OOF medio005=0,873877;007=0,877194; delta=+0,003317. Brecha media disminuye0,003237.
- Histórico137:005=0,873382;007=0,877822. Este resultado también participó en la puerta de selección.
- Público confirmado por capturas:005=0,88111;007=0,88888; diferencia=+0,00777. No asumir que la fórmula pública sea F1 macro: el nombre no aparece en la captura.
- Diagnóstico por clase en3313: negativo0,8246, positivo0,8215, neutral0,9882. El margen de mejora sigue en positivo/negativo.

Las alternativas global13, mid_bucket y edge_bigrams de007b empeoraron. Los canales simples de negación/contraste de007 tampoco pasaron. No repetirlos sin una diferencia concreta en la hipótesis.

## Precauciones sobre la interpretación

La candidata falló el umbral original de+0,005 por semilla. Después se adoptó otro protocolo de comparación con nuevas semillas y umbral medio+0,002. Se fijó antes de ejecutar esa segunda comparación, pero forma parte de una secuencia adaptativa motivada por la primera: no equivale a una prueba externa nueva.

Los18 folds ganados reutilizan las mismas filas entre semillas y tienen entrenamientos solapados. No justificar significancia mediante un test de signos que los trate como18 ensayos independientes. Las victorias son evidencia descriptiva de estabilidad, no certeza estadística. El valor p mencionado en un comentario del código no sustenta por sí solo esa conclusión.

El score público tampoco prueba qué mecanismo produjo la mejora. No extrapolar que una nueva mejora local de0,003 produzca otros0,00777 públicos. La distancia a0,90 en la escala pública es0,01112, no una garantía de estar a un número conocido de reseñas: no se ha verificado la métrica oficial.

## Propuesta acotada para008 — no ejecutada

Usar007 como referencia. Ronda inicial de cinco alternativas en total, con las siguientes hipótesis fijadas antes de entrenar:

1. **Referencia007 exacta**, para comprobar reproducción.
2. **Reducir el peso del bloque third a0,5** después del TF-IDF y renormalizar todo el vector. El vocabulario e IDF se aprenden dentro de cada fold. Comparar con el mismo procedimiento a peso1 como control técnico. Esto permite comprobar si la posición global ayuda pero tiene demasiado peso.
3. **Aumentar el peso third a1,5**, bajo las mismas condiciones. No afirmar de antemano qué dirección es mejor. Esta prueba se concentra en un factor, sin cambiar C, vocabulario ni segmentación simultáneamente.
4. **Posición continua suave de tres regiones**: en vez de asignar bruscamente cada palabra a un único tercio, repartir su contribución entre las dos regiones vecinas con pesos triangulares. Los puntos de referencia se fijan de antemano. Debe ser una representación de pesos fraccionarios en matriz dispersa; repetir tokens es una aproximación distinta y no debe presentarse como idéntica. Mantener un control de normalización y escala. Es una hipótesis sobre robustez a cambios pequeños de longitud, no un beneficio comprobado.
5. **Identidad del conector de cada cláusula como rasgo**, con un canal pequeño y peso fijo previamente declarado: aprender asociaciones palabra×conector, distinguiendo por ejemplo `aunque` de `por_cierto`. Esto difiere del canal anterior que agregaba todo lo posterior a contraste. Conservar siempre el texto global; no decidir etiquetas mediante conectores. Considerarla la alternativa de mayor riesgo de especificidad y memoria, y limitar frecuencia/vocabulario dentro del fold.

Si se prioriza rapidez, ejecutar solo referencia y las dos ponderaciones primero. Las otras dos necesitan implementación y controles adicionales; no introducirlas después de ver los resultados como si pertenecieran a un plan inalterado. La variante suave debe probar por separado si cambia simultáneamente la transformación TF logarítmica por necesitar pesos fraccionarios.

No volver de entrada a stacking, trigramas globales, bigramas posicionales generalizados o caracteres. La evidencia actual no los prioriza. Tampoco relabelar, eliminar observaciones difíciles ni usar reglas rígidas de polaridad.

## Criterio de trabajo

Registrar presupuesto, candidatos, tolerancias y criterio de rechazo antes de ejecutar. Comparar contra007 en los mismos folds, registrar diferencias emparejadas, F1 macro por clase y brecha. El propósito no es encontrar una semilla favorable. Si todos esos scores se usan para elegir, denominarlos selección, no prueba independiente.

Una sola evaluación de seguimiento del candidato congelado puede aportar estabilidad, pero si reutiliza train históricamente explorado debe declararlo. No bajar el umbral tras un fallo y seguir cambiando semillas hasta aprobar. Si no hay evidencia suficiente, conservar007.

Todas las transformaciones aprendidas, incluido vocabulario para ponderar bloques, selección de rasgos y cualquier IDF, deben ajustarse dentro del fold. Usar matrices dispersas. No tocar la prueba local antigua ni aprender de eval. No usar consultas del leaderboard como mecanismo de selección.

Este turno solo revisa y propone008; no se entrenó ni generó un nuevo envío. Se conservan los artefactos anteriores y el resultado público posterior se registra aparte en resultado_publico_007.json.
