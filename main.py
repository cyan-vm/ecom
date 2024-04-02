import nltk
import string

# Gramática libre de contexto

# S -> NP VP: (S) consists of a noun phrase (NP) followed by a verb phrase (VP).
# NP -> Det N | Det N PP: noun phrase (NP) can be composed of a determiner (Det) followed by a noun (N), or a determiner followed by a noun and a prepositional phrase (PP).
# VP -> V NP | VP PP: This rule specifies that a verb phrase (VP) can be a verb (V) followed by a noun phrase (NP), or a verb phrase followed by a prepositional phrase (PP).
# PP -> P NP: This rule indicates that a prepositional phrase (PP) consists of a preposition (P) followed by a noun phrase (NP)self.
# gramatica = nltk.CFG.fromstring("""
#     S -> NP VP                 
#     NP -> Det N | Det N PP
#     VP -> V NP | VP PP
#     PP -> P NP
#     Det -> 'el' | 'la' | 'un' | 'una' | 'a'
#     N -> 'gato' | 'pajaro' | 'Saul' | 'Cristhian' | 'Alondra :)' | 'tigre' | 'cotorro'
#     V -> 'observa' | 'come' | 'adora'
#     P -> 'yo' | 'tu' | 'el' | 'ella' | 'nosotros' | 'nosotras' | 'nos'
# """)

gramatica = nltk.CFG.fromstring("""
    S -> NP VP                 
    NP -> Det N | Det N PP
    VP -> V NP | VP PP
    PP -> P NP
    Det -> 'el' | 'la' | 'un' | 'una' | 'a'
    N -> 'gato' | 'pajaro' | 'Saul' | 'Cristhian' | 'Alondra :)' | 'tigre' | 'cotorro' | 'rapidez'
    V -> 'observa' | 'come' | 'adora'
    P -> 'yo' | 'tu' | 'el' | 'ella' | 'nosotros' | 'nosotras' | 'nos' | 'con'
""")




# Crear un parser con la gramática definida
parser = nltk.ChartParser(gramatica)

# Pedir al usuario que ingrese una oración para analizar
oracion = "el tigre come un cotorro con rapidez"
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
