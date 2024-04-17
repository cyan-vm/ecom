import nltk
import string

# Definir la gramática con información sobre el tiempo verbal
gramatica = nltk.CFG.fromstring(
 """
    O -> SN SV
    SN -> Det N Adj | Det Adj N PP | Det N | Det N PP
    SV -> V_presente_singular SN | V_presente_plural SN | V_presente_singular SN PP | V_presente_plural SN PP | SV PP | V_pasado_singular SN | V_pasado_plural SN | V_pasado_singular SN PP | V_pasado_plural SN PP | V_futuro_singular SN | V_futuro_plural SN | V_futuro_singular SN PP | V_futuro_plural SN PP
    PP -> P SN | P N
    Det -> 'el' | 'la' | 'un' | 'una' | 'al' | 'los' | 'las'
    Adj -> 'hermoso' | 'rápido' | 'grande' | 'pequeño'| 'hermosos'| 'rápidos' | 'grandes' | 'pequeños'| 'hermosas'| 'rápidas' | 'grandes' | 'pequeñas' 
    N -> 'gato' | 'pajaro' | 'Saul' | | 'cristhian' | | 'Alondra' | 'perro' | 'elefante' | 'aguila' | 'tigre' | 'caballo' | 'elegancia'| 'tigres'| 'pajaros'| 'perros'| 'elefantes'| 'aguilas'| 'caballos'| 'gatos'
    V_presente_singular -> 'observa' | 'come' | 'adora'
    V_presente_plural -> 'observan' | 'comen' | 'adoran'
    V_pasado_singular -> 'observó' | 'comió' | 'adoró'
    V_pasado_plural -> 'observaron' | 'comieron' | 'adoraron'
    V_futuro_singular -> 'observará' | 'comerá' | 'adorará'
    V_futuro_plural -> 'observarán' | 'comerán' | 'adorarán'
    P -> 'con'
  """
)

analizador = nltk.ChartParser(gramatica)

# oracion = "¿los hermosos tigres observaron el grande caballo?"

# oracion = 'el rápido elefante observo al hermoso tigre con elegancia'

# oracion = 'el rápido gato observó al hermoso tigre con elegancia'# oracion = 'el rápido gato observa al hermoso tigre con elegancia'
oracion = "¿ el gato rápido observa al tigre hermoso con elegancia ?"

# oracion = "el hermoso gato observó al pajaro con rapidez"

esInterrogacion = False

# Verificar si la oración termina con un signo de interrogación o empieza con signo de interrogacion
if oracion.startswith("¿") and oracion.endswith("?"):
    # Quitar el signo de interrogación
    oracion = oracion[1:]
    oracion = oracion[:-1]
    esInterrogacion = True

# Quitar puntuación y tokenizar la oración
traductor = str.maketrans("", "", string.punctuation)
oracion_sin_puntuacion = oracion.translate(traductor)
simbolos = oracion_sin_puntuacion.split()

try:
    # Intentar analizar la oración con la gramática definida
    arbol_parseo = list(analizador.parse(simbolos))

    # Mostrar las producciones gramaticales
    print("\nProducciones de la gramática:")
    for produccion in gramatica.productions():
        print(produccion)

    # Mostrar el árbol de análisis
    print("\nÁrbol de parseo:")
    arbol_parseo[0].pretty_print()

    # Encontrar el verbo en el árbol de análisis y determinar su tiempo y número
    tiempo_verbo = ""
    verbo_sp = ""
    for subarbol in arbol_parseo[0].subtrees():
        if subarbol.label() == "V_presente_singular":
            tiempo_verbo = "presente"
            verbo_sp = "singular"
        elif subarbol.label() == "V_presente_plural":
            tiempo_verbo = "presente"
            verbo_sp = "plural"
        elif subarbol.label() == "V_pasado_singular":
            tiempo_verbo = "pasado"
            verbo_sp = "singular"
        elif subarbol.label() == "V_pasado_plural":
            tiempo_verbo = "pasado"
            verbo_sp = "plural"
        elif subarbol.label() == "V_futuro_singular":
            tiempo_verbo = "futuro"
            verbo_sp = "singular"
        elif subarbol.label() == "V_futuro_plural":
            tiempo_verbo = "futuro"
            verbo_sp = "plural"

    if tiempo_verbo:
        print(
            f"\nEl verbo utilizado está en tiempo {tiempo_verbo} y es {verbo_sp}."
        )
    else:
        print("\nNo se pudo determinar el tiempo verbal y número del verbo.")

    print("\nLa oración cumple con las reglas gramaticales.")

except IndexError:
    print("\nLa oración no cumple con las reglas gramaticales.")

if esInterrogacion:
    print("La oración es una pregunta")
