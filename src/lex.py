import re

NEG_PH = ['no vale lo que cuesta', 'para nada lo que esperaba', 'no pare de tener problemas',
          'dolor de cabeza', 'error de compra', 'tire el dinero', 'ni regalado', 'no volveria',
          'me arrepiento', 'arrepentid', 'decepcion', 'defraud', 'se dano', 'mal hech', 'mal termin',
          'no sabria recomendar', 'no le recomiendo', 'no lo recomiendo', 'no la recomiendo',
          'por debajo', 'deja mucho que desear', 'dejo mucho que desear', 'no me convencio',
          'no funciono', 'no sirve', 'me fallo', 'fallo']

POS_PH = ['ni una sola queja', 'ninguna queja', 'sin quejas', 'cinco estrellas', 'vale la pena',
          'valio la pena', 'vale totalmente', 'ojos cerrados', 'de sobra', 'tal como esperaba',
          'lo que estaba buscando', 'justo lo que', 'todo un acierto', 'de maravilla', 'supero',
          'me alegra', 'feliz', 'encantad', 'me encanto', 'me sorprendio', 'valio cada', 'bien hech',
          'bien constru', 'bien logr', 'bien termin', 'por encima de la media', 'volveria a comprar',
          'vida mas facil', 'funciona como debe', 'para bien']

POS_W = set('resistente acierto eficiente impecable veloz funcional confiable '
            'sobresaliente potente decente superior notable formidable excelente espectacular durable '
            'agradable cumplidor cumplidora buenisima buenisimo comodo comoda comodisima comodisimo '
            'rapidisima rapidisimo rapido rapida optimo optima precisa preciso correcta correcto solida '
            'solido ligera ligero robusta robusto lograda logrado nitida nitido estupenda estupendo '
            'redonda redondo fiable buena bueno recomiendo practica practico silencioso silenciosa '
            'firme estable'.split())

NEG_W = set('terrible torpe pobre debil molesto molesta deficiente mediocre '
            'desagradable mala malo endeble fragil inestable inconsistente olvidable peor incumplidor '
            'incumplidora ruidosa ruidoso delicada delicado imprecisa impreciso lentisima lentisimo '
            'lenta lento chapucera chapucero flojo floja pesima pesimo incomoda incomodo malisima '
            'malisimo ordinaria ordinario defectuosa defectuoso limitado limitada aparatosa '
            'aparatoso'.split())

def polarity(body):
    s, b = 0, ' ' + body + ' '
    for p in NEG_PH:
        if p in b:
            s -= 1
            b = b.replace(p, ' ')
    for p in POS_PH:
        if p in b:
            s += 1
            b = b.replace(p, ' ')
    for w in re.findall(r'\w+', b):
        if w in POS_W:
            s += 1
        elif w in NEG_W:
            s -= 1
    return s