import nltk
import string

# Gramática libre de contexto

# gramatica = nltk.CFG.fromstring("""
#     S -> NP VP                 
#     NP -> Det N | Det N PP
#     VP -> V NP | VP PP
#     PP -> P NP
#     Det -> 'el' | 'la' | 'un' | 'una' | 'a'
#     N -> 'gato' | 'pajaro' | 'Saul' | 'Cristhian' | 'Alondra :)' | 'tigre' | 'cotorro' | 'rapidez'
#     V -> 'observa' | 'come' | 'adora'
#     P -> 'yo' | 'tu' | 'el' | 'ella' | 'nosotros' | 'nosotras' | 'nos' | 'con' 
# """)

# gramatica = nltk.CFG.fromstring("""
#     S -> NP VP
#     NP -> Det N | Det N PP
#     VP -> V NP | VP PP
#     PP -> P NP
#     Det -> 'el' | 'la' | 'un' | 'una' | 'al'
#     N -> 'gato' | 'pajaro' | 'Saul' | 'Christian' | 'Alondra' | 'tigre' | 'cotorro' | 'rapidez'
#     V -> 'observa' | 'come' | 'adora'
#     P -> 'con'
# """)

gramatica = nltk.CFG.fromstring("""
    S -> NP VP
    NP -> Det N | Det N PP
    VP -> V NP | V NP PP | VP PP
    PP -> P NP | P N
    Det -> 'el' | 'la' | 'un' | 'una' | 'al'
    N -> 'gato' | 'pajaro' | 'Saul' | 'Christian' | 'Alondra' | 'tigre' | 'cotorro' | 'rapidez'
    V -> 'observa' | 'come' | 'adora'
    P -> 'con'
""")

# Crear un parser con la gramática definida
parser = nltk.ChartParser(gramatica)

# Pedir al usuario que ingrese una oración para analizar
# oracion = "el tigre come un cotorro"
oracion = "el gato observa al pajaro con rapidez"
# oracion = "el tigre come un cotorro con Alondra"

# oracion = "el tigre come un cotorro que adora Saul"



# Eliminar signos de puntuación y tokenizar la oración
translator = str.maketrans("", "", string.punctuation)
oracion_sin_puntuacion = oracion.translate(translator)
tokens = oracion_sin_puntuacion.split()

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
