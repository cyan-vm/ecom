import os
import nltk

# Definir la gramática en la forma BNF
gramatica = nltk.CFG.fromstring("""
O -> SN SV
SN -> Det Nom | Det Adj Nom | Det Adj Adj Nom | SN SP
SV -> V | V SN | V SN SN | V SP | V SP SN | V SP
SP -> Prep SN | Prep NProp
Nom -> N | Adj Nom
Det -> 'el' | 'la' | 'los' | 'las' | 'un' | 'una' | 'unos' | 'unas' | 'este' | 'esta' | 'estos' | 'estas' | 'ese' | 'esa' | 'esos' | 'esas' | 'mi' | 'tu' | 'su' | 'nuestro' | 'vuestro' | 'sus' | 'algunos' | 'algunas' | 'todas' | 'todos' | 'solo' | 'Solo' | 'El' | 'La' | 'Los' | 'Las' | 'Un' | 'Una' | 'Unos' | 'Unas' | 'Este' | 'Esta' | 'Estos' | 'Estas' | 'Ese' | 'Esa' | 'Esos' | 'Esas' | 'Mi' | 'Tu' | 'Su' | 'Nuestro' | 'Vuestro' | 'Sus' | 'Algunos' | 'Algunas' | 'Todas' | 'Todos'
N -> 'joven' | 'chico' | 'chica' | 'mandarinas' | 'lápiz' | 'perro' | 'gato' | 'casa' | 'temprano' | 'mesa' | 'silla' | 'niño' | 'niña' | 'mujer' | 'hombre' | 'computadora' | 'teléfono' | 'libro' | 'auto' | 'bicicleta' | 'ciudad' | 'país' | 'amigo' | 'amiga' | 'familia' | 'comida' | 'estudia' | 'mañanas' | 'trabaja' | 'tardes' | 'es' | 'apellido' | 'muy' | 'común' | 'región'
NProp -> 'Azeneth' | 'Iveth' | 'Rosas' | 'María' | 'José' | 'Juan' | 'Ana' | 'Luis' | 'Carlos' | 'Laura' | 'David' | 'Carmen' | 'Manuel' | 'Rosa' | 'Pedro' | 'Patricia' | 'Antonio' | 'Isabel' | 'García' | 'Martínez' | 'Pérez' | 'González'
Adj -> 'hermoso' | 'pequeña' | 'plata' | 'listo' | 'bueno' | 'bien' | 'fiel' | 'fácil' | 'frío' | 'feo' | 'claro' | 'barato' | 'débil' | 'divertido' | 'frito' | 'enfermo' | 'libre' | 'limpio' | 'sucio' | 'pequeño' | 'lleno' | 'gastado' | 'largo' | 'malo' | 'feliz' | 'atrevido' | 'simpático' | 'injusto' | 'paciente' | 'aburrido' | 'cuidado' | 'humano' | 'elegante' | 'cerrado' | 'egoísta' | 'alegre' | 'triste' | 'tranquilo' | 'nervioso'
V -> VPS | VPP | VPsS | VPsP | VFS | VFP
VPS -> 'observa' | 'come' | 'adora' | 'pinta' | 'pasear'
VPP -> 'observan' | 'comen' | 'adoran' | 'cantaban' | 'salen'
VPsS -> 'observó' | 'comió' | 'adoró'
VPsP -> 'observaron' | 'comieron' | 'adoraron'
VFS -> 'observará' | 'comerá' | 'adorará'
VFP -> 'observarán' | 'comerán' | 'adorarán'
Prep -> 'a' | 'ante' | 'bajo' | 'contra' | 'de' | 'desde' | 'en' | 'entre' | 'hacia' | 'hasta' | 'para' | 'por' | 'y' | 'según' | 'durante' | 'mediante' | 'sobre' | 'sin' | 'tras' | 'pro' | 'con'
""")

# Crear un parser con la gramática definida
parser = nltk.ChartParser(gramatica)

while True:
  # Pedir al usuario que ingrese una oración para analizar
  oracion = input("Ingresa una oración (o escribe 'salir' para salir): ")

  if oracion.lower() == 'salir':
    break

  # Tokenizar la oración
  tokens = oracion.split()

  try:
    # Intentar analizar la oración con la gramática definida
    arbol_parseo = list(parser.parse(tokens))

    # Mostrar las producciones de la gramática
    print("\nProducciones de la gramática:")
    for produccion in gramatica.productions():
      print(produccion)
      

    # Mostrar el árbol de parseo
    print("\nÁrbol de parseo:")
    arbol_parseo[0].pretty_print()

    print("\nLa oración cumple con las reglas gramaticales.")

  except IndexError:
    print("\nLa oración no cumple con las reglas gramaticales.")
