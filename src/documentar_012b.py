"""Registra en README y uso_ia la ejecución cerrada012b."""
from pathlib import Path
import json
import pandas as pd
r=json.loads(Path('reports/registro_012b.json').read_text())
cv=pd.read_csv('reports/resultados_012b.csv');m=cv[cv.seed.isin(r['plan']['seeds'])].groupby('name')[['accuracy','f1','gap']].mean()
heading='## Continuación012b, 9 de octubre de2026'
text='\n'+heading+'\n\n'
text+='La captura011 registra score0,89888. La entrega011 proviene de producir_011.py: mismos coeficientes que010 y ajuste de0,2 a los márgenes positivo/negativo; cambia4 predicciones. Se verificaron hashes y reproducción. La nueva referencia incorpora la regla completa en predict(). No se modifica el .joblib anterior, que guarda la base sin corrección.\n\n'
text+='Se investigaron y ejecutaron especialistas binarios de polaridad C=1/C=2, clasificación por pares y calibración sigmoide con OOF interna agrupada. Accuracy principal, F1 macro control.45 ajustes externos de pipeline en8803/9901/11003, más comparación137 solo para elegibles. Los ajustes internos del calibrador incluyen vectorizador y clasificador. Datos de desarrollo ya explorados; no prueba nueva.\n\n'
text+='El criterio se fijó antes de entrenar: delta accuracy>=0,001; al menos7/9 victorias; F1 medio no inferior; aumento de brecha<=0,005; no perder accuracy ni F1 en137. Es un criterio nuevo para esta ronda, distinto de011b, no una reclasificación retrospectiva de sus resultados.\n\n'
text+='| Modelo | Accuracy media | F1 macro medio | Brecha accuracy |\n|---|---:|---:|---:|\n'
for name,row in m.sort_values('accuracy',ascending=False).iterrows():text+=f'| {name} | {row.accuracy:.6f} | {row.f1:.6f} | {row.gap:.6f} |\n'
text+='\n'+('Se generó submissions/submission_012.csv con '+r['winner']+'. No enviado a la plataforma.' if r['submission_created'] else r['decision'])+'\n\n'
text+='No se infiere un techo absoluto de ML clásico ni se promete0,90 público. Tampoco la frecuencia de neutral demuestra que el descuento anterior sea una corrección probabilística de prior. La documentación previa afirma selección anidada de ese descuento, pero sus tablas/código de semillas no se localizaron en esta revisión. Los resultados012b sí quedan guardados por fold y ejemplo.\n\n'
text+='Reproducción: `.venv/bin/python -m src.experimento_012b` (protege registro y destinos existentes), `.venv/bin/python -m src.verificar_012b` y `.venv/bin/python -m src.informe_012b`. Ver reports/resumen_012b.md, reports/investigacion_012b.md, reports/verificacion_012b.json y notebooks/12b_revision_y_modelo.ipynb.\n'
for filename in ['README.md','uso_ia.md']:
    p=Path(filename)
    if heading in p.read_text():raise FileExistsError(filename)
    intro=''
    if filename=='uso_ia.md':intro='\nSolicitud del usuario: revisar submission011, investigar mejoras e iniciar012 para aumentar desempeño, seguida de «continue». El asistente auditó, investigó documentación oficial, implementó y evaluó alternativas sin datos externos, nuevas etiquetas ni acceso a Kaggle.\n'
    with p.open('a') as f:f.write(intro+text)
print(text)
