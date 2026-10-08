# Revisión del auditor antes de la versión 005

La captura del usuario reporta 0,86111 para submission_004.csv: aumento de 0,01778 sobre el público 003. Se registra como contexto posterior al entrenamiento, sin asumir el nombre de la métrica ni usar el score para escoger parámetros.

## Cambios comprobados

`features_004.py` parametriza el analizador de 003 y conserva el módulo original. `experimento_004.py` compara ocho configuraciones. La ganadora `ng2` reduce de 1–3 a 1–2 los n-gramas de cláusulas con posición; conserva bigramas globales, segmentación por conectores, tratamiento de epílogo, canal de caracteres de peso 0,5, min_df=3 y LinearSVC C=2.

La tabla ejecutada muestra F1 de CV 0,851422 y F1 OOF 0,850578. La réplica reproduce 003. Es evidencia a favor de simplificar la representación, no una demostración de la causa exacta de los errores. Entrenamiento sigue en 0,999951 y la brecha de CV en 0,148529.

## Limitación metodológica que corregir

En `experimento_004.py`, se evalúan todas las alternativas en los antiguos folds externos y se utiliza `oof_f1` para seleccionar la ganadora. Por tanto, aunque cada predicción se hizo fuera del entrenamiento de esa fila, el agregado OOF participó en la selección. No debe presentarse como evaluación independiente del procedimiento. El informe del auditor reconoce que los folds ya se habían visto; además de esa reutilización histórica, hay selección directa sobre ellos en 004.

La 005 repetirá la selección exclusivamente dentro del entrenamiento de cada fold externo. El resultado agregado externo no se usará para cambiar hiperparámetros. Esta separación tampoco borra la adaptación acumulada al desarrollo ni crea una prueba nueva.

## Ronda 005 fijada antes de ejecutar

Seis configuraciones: referencia 004; solo unigramas posicionales; min_df=10; vocabulario discursivo máximo 30000; unigramas sin caracteres; bigramas sin caracteres. Se mantiene C=2 y el resto del diseño. No se cambia de clasificador ni se añaden reglas basadas en eval.

Regla: usar F1 macro con las tres etiquetas; limitar brecha a la de referencia más 0,002; entre alternativas a 0,003 del mejor F1 elegible, preferir menor brecha. Exigir mejora mínima de 0,002 sobre la referencia para escoger una versión diferente. Guardar la regla y resultados completos. La tolerancia es una decisión práctica, no significancia estadística.

CV anidada 3 externa x 3 interna sobre desarrollo. Un proceso y un hilo numérico para controlar memoria. El ajuste captura ConvergenceWarning como error y verifica iteraciones. Los modelos se clonan y cada vectorizador se ajusta dentro del fold. La antigua prueba local no se vuelve a evaluar. Después de cerrar selección, el modelo de producción se ajusta con todo train.

Los artefactos previos quedan preservados mediante SHA-256. La nueva versión se numera 005; con la información aportada constan tres envíos reales (002, 003, 004), no cuatro. El número del archivo no demuestra cuántos envíos ha hecho el usuario.
