from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
import sys
path=Path('notebooks/01_modelo_inicial.ipynb')
nb=nbformat.read(path,as_version=4)
km=KernelManager(kernel_name='python3')
km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client=NotebookClient(nb,timeout=7200,km=km,resources={'metadata':{'path':str(Path.cwd())}})
try:
    client.execute()
finally:
    nbformat.write(nb,path)
print('Notebook ejecutado y guardado.')
