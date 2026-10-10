import numpy as np
from src.clausulas import clauses
from src.lex import polarity

ADD = {'por otro lado', 'de paso', 'por cierto', 'ademas'}
CON = {'al final', 'aun asi', 'con todo', 'pero', 'a fin de cuentas', 'sin embargo',
       'al final de cuentas', 'aunque'}


def ctype(c):
    if c in ADD:
        return 'add'
    if c in CON:
        return 'con'
    return c.replace(' ', '_') or 'none'


def rule_tokens(text):
    cl = clauses(text)
    pols = [(k, c, int(np.sign(polarity(b)))) for k, (c, b) in enumerate(cl)]
    P = [(k, c, p) for k, c, p in pols if p != 0]
    if not P:
        yield 'R:nopol'
        return
    sig = ''.join('P' if p > 0 else 'N' for _, _, p in P)
    k, c, p = P[-1]
    L = 'P' if p > 0 else 'N'
    after = str(int(k < len(cl) - 1))
    ct = ctype(c)
    yield 'R:last=' + L
    yield 'R:last=' + L + '|ct=' + ct
    yield 'R:last=' + L + '|ct=' + ct + '|after=' + after
    yield 'R:first=' + sig[0]
    yield 'R:sig=' + sig[-3:]
    yield 'R:sig=' + sig[-3:] + '|ct=' + ct + '|after=' + after