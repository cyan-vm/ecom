import nltk
import string

gramatica = nltk.CFG.fromstring("""
    S -> NP VP
    NP -> Det Adj N | Det Adj N PP | Det N | Det N PP
    VP -> V NP | V NP PP | VP PP
    PP -> P NP | P N
    Det -> 'el' | 'la' | 'un' | 'una' | 'al'
    Adj -> 'hermoso' | 'rápido' | 'grande' | 'pequeño'
    N -> 'gato' | 'pajaro' | 'Saul' | 'Christian' | 'Alondra' | 'tigre' | 'cotorro' | 'rapidez'
    V -> 'observa' | 'come' | 'adora' | 'observó'
    P -> 'con'
""")

# Crear un parser con la gramática definida
parser = nltk.ChartParser(gramatica)

# Pedir al usuario que ingrese una oración para analizar
# oracion = "el tigre come un cotorro"
oracion = "el hermoso gato observa al pajaro con rapidez"
# oracion = "el hermoso gato observó al pajaro con rapidez"


# oracion = "el tigre come un cotorro con Alondra"

# oracion = "el tigre come un cotorro que adora Saul"

# Eliminar signos de puntuación y tokenizar la oración
translator = str.maketrans("", "", string.punctuation)
oracion_sin_puntuacion = oracion.translate(translator)
tokens = oracion_sin_puntuacion.split()

try:
    # Intentar analizar la oración con la gramática definida
    arbol_parseo = list(parser.parse(tokens))

    # Función para determinar el tiempo verbal de un verbo
# Función para determinar el tiempo verbal de un verbo
    def determinar_tiempo_verbal(arbol):
        for nodo in arbol.subtrees():
            if nodo.label() == 'V':
                if 'observa' in nodo[0] or 'come' in nodo[0] or 'adora' in nodo[0]:
                    return "Presente"
                elif 'observó' in nodo[0]:
                    return "Pasado"
                # Agregar más condiciones para otros verbos y contextos
                else:
                    return "Desconocido"
        return "N/A"


    # Mostrar las producciones de la gramática
    print("\nProducciones de la gramática:")
    for produccion in gramatica.productions():
        print(produccion)

    # Mostrar el árbol de parseo
    print("\nÁrbol de parseo:")
    arbol_parseo[0].pretty_print()

    tiempo_verbal = determinar_tiempo_verbal(arbol_parseo[0])
    print(f"\nTiempo verbal identificado: {tiempo_verbal}")

except IndexError:
    print("\nLa oración no cumple con las reglas gramaticales.")
