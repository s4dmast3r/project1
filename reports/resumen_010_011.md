# Rondas 010 y 011 — la 009 es un plateau estable

Fecha: 8 de octubre de 2026. Métrica exclusiva: F1 macro (`zero_division=0`).
Público confirmado 009 = 0,89555 (faltan 0,00445 para 0,90).

## Qué se probó (criterios fijados antes de ejecutar)

### 010 — barrido de C alrededor de la 009
Candidatas: ref009 (L1 C=1), l1_c05 (C=0.5), l1_c15 (C=1.5), l1_c2 (C=2.0) y
l2_c1 (L2 C=1, control). Semillas nuevas 5003/6007/7001.

| Candidata | Δ F1 OOF medio | Folds ganados | Δ brecha | Veredicto |
|---|---:|---:|---:|---|
| l1_c05 | −0.0174 | 0/9 | −0.0162 | peor (subajuste) |
| l1_c15 | +0.0017 | 8/9 | +0.0179 | brecha peor |
| l1_c2 | +0.0019 | 6/9 | +0.0290 | brecha peor |
| l2_c1 | −0.0271 | 0/9 | +0.0662 | mucho peor |

Conclusión: C=1 es el óptimo. El control **L2 C=1** (que retrocede a OOF ~0.873 y
brecha 0.12) aísla que la ganancia de la 009 proviene de la **penalización L1**
(dispersión), no del valor de C.

### Complementariedad con 008b (NB-SVM corregido)
Evaluados 009 y NBRatioSVM en los mismos folds (8009/9001/1103): 009 = 0.9015,
NBRatioSVM = 0.8926; solapan el 68 % de los errores de 009 y el voto por rangos cae
a 0.8897. **La combinación no ayuda**; el modelo NB es más débil y arrastra al L1.

### 011 — re-ensayar bajo L1 rasgos que fallaron bajo L2
Candidatas: l1_global13 (trigramas globales), l1_edgebig (bigramas de borde),
l1_charwb (canal char_wb 2-6), l1_min2 (min_df=2). Semillas nuevas 2203/3307/4409.

| Candidata | Δ F1 OOF medio | Folds ganados | Veredicto |
|---|---:|---:|---|
| l1_global13 | −0.00018 | 4/9 | neutro |
| l1_edgebig | −0.00132 | 1/9 | neutro/peor |
| l1_charwb | +0.00046 | 5/9 | neutro |
| l1_min2 | −0.00049 | 3/9 | neutro |

Ninguna supera el umbral (≥0.003). Bajo L1, esos rasgos adicionales no aportan señal
más allá del ruido entre folds.

### Calibración de umbral (S10)
Sesgo por clase sobre los scores L1 en 137 (selección anidada): media +0.0016 sobre
la base 0.89909, con sesgos inconsistentes entre folds. Es ruido, no un sesgo real
de decisión.

## Decisión

**Conservar 009** (L1 C=1 sobre las características 007). Ninguna candidata de las
rondas 010/011 cumplió las cuatro puertas pre-registradas; no se generaron
`submission_010.csv` ni `submission_011.csv`. Los artefactos 001–009 permanecen
intactos.

## Interpretación honesta

La 009 está en un plateau: F1 macro OOF ~0.90 y público 0.89555. El salto de la 009
(+0.024 OOF local, +0.0067 público) provino de corregir un sobreajuste extremo con
L1; ese margen ya se consumió. Los 0.00445 restantes hasta 0.90 en la escala pública
no son alcanzables con las estrategias probadas (C, rasgos, ensembles, calibración):
todas quedan dentro de ±0.002 del plateau.

No se puede garantizar 0.90. El techo observable con ML tradicional y este corpus
(plantillado, con reseñas de aspecto mixto difícil de separar) está en torno a
0.895–0.90 en la escala pública no verificada. Reemplazar 009 por un cambio de
ruido arriesgaría repetir la caída pública de la 006.
