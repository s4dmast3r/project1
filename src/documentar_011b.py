"""Añade el resultado final y la solicitud a la documentación del proyecto."""
from pathlib import Path
import json
import pandas as pd
r=json.loads(Path('reports/registro_011b.json').read_text())
cv=pd.read_csv('reports/resultados_011b.csv');m=cv[cv.seed.isin(r['plan']['seeds'])].groupby('name')[['accuracy','f1','gap']].mean()
heading='## Continuación011b, 9 de octubre de2026'
text='\n'+heading+'\n\n'
text+='La captura del usuario confirma público010=0,89777. La entrega010 proviene del experimento013 (L1 C=2); los antiguos registros010/011 son rondas distintas sin aprobación. Se verificaron fuentes/modelo/CSV de010.\n\n'
text+='El usuario confirmó en este chat: «Accuracy como principal y F1 macro como control». Se consultaron fuentes primarias de scikit-learn y COLING, y se implementaron cinco alternativas: binary_idf, binary_noidf, tf_noidf, connector_l1 y logistic_l1. Se reutilizan particiones8803/9901/11003 para comparación con010; no son datos nuevos. Todos los aprendizajes quedan dentro del fold. No se accedió a Kaggle ni se añadieron datos externos.\n\n'
text+='| Modelo | Accuracy media | F1 macro medio | Brecha accuracy |\n|---|---:|---:|---:|\n'
for name,row in m.sort_values('accuracy',ascending=False).iterrows():text+=f'| {name} | {row.accuracy:.6f} | {row.f1:.6f} | {row.gap:.6f} |\n'
text+='\n'+('Se generó submissions/submission_011.csv con '+r['winner']+'. No enviado a la plataforma.' if r['submission_created'] else r['decision'])+'\n\n'
text+='Los resultados son de selección adaptativa acumulada, no evaluación externa independiente. No se concluye un techo absoluto de rendimiento a partir de esta ronda ni se promete0,90 público. Se conservan los experimentos previos.\n\n'
text+='Reproducción: `.venv/bin/python -m src.experimento_011b` (bloquea sobrescritura del registro existente), `.venv/bin/python -m src.verificar_011b` y `.venv/bin/python -m src.informe_011b`. Ver reports/resumen_011b.md, reports/investigacion_011b.md, reports/verificacion_011b.json y notebooks/11b_revision_y_modelo.ipynb.\n'
for filename in ['README.md','uso_ia.md']:
    p=Path(filename)
    if heading in p.read_text():raise FileExistsError('La ronda ya está documentada: '+filename)
    intro=''
    if filename=='uso_ia.md':intro='\nSolicitud: «quiero que revises la entrega010, y que investigues y que saques la versión011 [...] para aumentar el desempeño». El asistente auditó, investigó, implementó y ejecutó una ronda acotada con criterios fijados antes de observar resultados.\n'
    with p.open('a') as f:f.write(intro+text)
print(text)
