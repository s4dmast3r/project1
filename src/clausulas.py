import re
import unicodedata

#Normalizar y limpiar el texto.
def clean(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(s).lower())
                   if not unicodedata.combining(c))

#Conectores:
CONN = (r'(eso si|aun asi|sin embargo|no obstante|con todo|por otro lado|por cierto|de paso|'
        r'ademas|aunque|pero|al final(?: de cuentas)?|a fin de cuentas|'
        r'y (?=(?:el|la|los|las|al|me|lo) ))')

SPLIT = re.compile(r'\b' + CONN + r'\b')

#Parte cada reseña en clausulas junto con el conector que las introduce. 
def clauses(text):
    out = []
    for sent in re.split(r'[.!?;]+', clean(text)):
        sent = sent.strip(' ,;')
        if not sent:
            continue
        parts = SPLIT.split(sent)
        first = parts[0].strip(' ,:')
        if first:
            out.append(('', first))
        # Fuera del if: una oración que empieza con conector ("Pero ...") conserva sus cláusulas.
        for k in range(1, len(parts), 2):
            body = parts[k + 1].strip(' ,:')
            if body:
                out.append((parts[k].strip(), body))
    return out
