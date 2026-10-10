# Ronda012b — auditoría011 y decisiones multiclase

Accuracy principal y F1 macro control por autorización del usuario. Score visible011: **0,89888**, frente a010:0,89777. Score012 desconocido.

## Qué cambió en011

Los coeficientes de011 son idénticos a010: LinearSVC L1 C=2 y las mismas características. El cambio es sumar0,2 a los márgenes positivo y negativo antes del argmax. Solo4 de3000 predicciones cambian. La subida pública es0,00111; no permite identificar los aciertos públicos ni demostrar un techo de rendimiento.

El artefacto modelo_011.joblib guarda el clasificador base sin esa regla: su predict() difiere del CSV en4filas. La regla se aplica en producir_011.py. La nueva referencia encapsula ambos componentes en predict() y usa nombres de clase. Los artefactos previos no se modifican. El script anterior sí reentrena la base, aunque el informe decía que no; se verificó igualdad de coeficientes.

## Métodos investigados e implementados

- `ref011`: reproducción de011 con descuento neutral0,2.
- `polarity_c1` y `polarity_c2`: conservan exactamente la puerta neutral011; un SVM binario decide positivo/negativo, entrenado solo con ejemplos polares del entrenamiento.
- `ovo_l1`: tres clasificadores entre pares de clases y combinación de decisiones, todos L1 C=2.
- `sigmoid`: calibración con predicciones internas fuera de fold; tres particiones internas agrupadas, vocabulario/IDF dentro de cada ajuste y refit final de la base. Sin descuento manual adicional.

Fuentes y protocolo: [investigacion_012b.md](investigacion_012b.md), [OneVsOneClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html) y [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html). No se añadieron datos externos ni sintéticos.

## Resultados locales de selección

| Candidata | Accuracy media OOF | F1 macro medio OOF | Brecha accuracy | Folds ganados |
|---|---:|---:|---:|---:|
| ref011 | 0.900243 | 0.904792 | 0.088056 | — |
| sigmoid | 0.900104 | 0.904595 | 0.088194 | 2/9 |
| polarity_c2 | 0.897014 | 0.901720 | 0.088316 | 0/9 |
| ovo_l1 | 0.895590 | 0.900167 | 0.089115 | 0/9 |
| polarity_c1 | 0.895208 | 0.899998 | 0.057222 | 1/9 |

Las medias OOF promedian los resultados de9600 predicciones de desarrollo por semilla. La brecha es accuracy de entrenamiento menos validación, promediada entre folds. Son las mismas particiones para todas las candidatas.

## Criterios y decisión

>=7/9 folds accuracy; delta medio accuracy >=0.001; delta F1 medio >=0; aumento brecha accuracy media <=0.005; no perder accuracy ni F1 en137.

Umbral de ganancia0.001 y tolerancia de brecha0.005 fijados antes de medir esta ronda; distintos de011b. No implican significación ni ausencia de sobreajuste.

Este es un protocolo nuevo, más sensible a mejoras pequeñas que011b; no se reclasifican retrospectivamente los resultados de esa ronda. Las condiciones se fijaron antes de medir012b y no se relajaron después.

Ganadora: **ninguna**. CSV creado: **False**.

Conservar011: ninguna candidata supera los criterios. No crear un envío equivalente ni bajar umbrales a posteriori.

## Controles y límites

Las9600 filas de desarrollo son las únicas usadas para seleccionar. El test local histórico no se reutiliza para decidir; eval se usa únicamente para auditar artefactos anteriores y exportar después de decidir. Grupos de texto normalizado disjuntos. Vocabulario, IDF, clasificadores y calibración aprendidos dentro del entrenamiento correspondiente; convergencia exigida. La calibración interna se rehace dentro de cada fold externo.

Hay selección adaptativa acumulada sobre este dataset: nuevas semillas no crean datos vírgenes. La CV interna evita fuga del fold externo al calibrador, pero no elimina el sesgo de elegir entre alternativas tras muchas rondas. Los nueve folds repetidos no son independientes. Una brecha menor no demuestra ausencia de sobreajuste y una tolerancia de0,005 no significa que ningún aumento sea aceptable universalmente.

El registro previo011 declara una calibración anidada de su descuento, pero no se localizaron tablas/código de esas semillas en esta revisión. La exportación011 sí se verificó, y la nueva comparación es reproducible. No se asume que frecuencia de clases justifique automáticamente descontar un margen SVM. No se promete0,90 público ni se concluye un techo absoluto del ML clásico.

## Reproducibilidad

Tiempo: 932.8 segundos. Artefactos anteriores intactos: True.

`.venv/bin/python -m src.experimento_012b` (protege destinos existentes), `.venv/bin/python -m src.verificar_012b` y `.venv/bin/python -m src.informe_012b`. El notebook analiza resultados guardados; no vuelve a entrenar. Ver plan, registro, criterios, folds y predicciones_oof con sufijo012b.
