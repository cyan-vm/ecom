import nltk
import string

# Defining the grammar with information about the verb tense and number
gramatica = nltk.CFG.fromstring(
  """
    S -> NP VP
    NP -> Det Adj N | Det Adj N PP | Det N | Det N PP
    VP -> V_presente_singular NP | V_presente_plural NP | V_presente_singular NP PP | V_presente_plural NP PP | VP PP | V_pasado_singular NP | V_pasado_plural NP | V_pasado_singular NP PP | V_pasado_plural NP PP | V_futuro_singular NP | V_futuro_plural NP | V_futuro_singular NP PP | V_futuro_plural NP PP
    PP -> P NP | P N
    Det -> 'el' | 'la' | 'un' | 'una' | 'al' | 'los' | 'las'
    Adj -> 'hermoso' | 'rápido' | 'grande' | 'pequeño'| 'hermosos'| 'rápidos' | 'grandes' | 'pequeños'| 'hermosas'| 'rápidas' | 'grandes' | 'pequeñas'
    N -> 'gato' | 'pajaro' | 'perro' | 'elefante' | 'aguila' | 'tigre' | 'caballo' | 'elegancia'| 'tigres'| 'pajaros'| 'perros'| 'elefantes'| 'aguilas'| 'caballos'| 'gatos'
    V_presente_singular -> 'observa' | 'come' | 'adora'
    V_presente_plural -> 'observan' | 'comen' | 'adoran'
    V_pasado_singular -> 'observó' | 'comió' | 'adoró'
    V_pasado_plural -> 'observaron' | 'comieron' | 'adoraron'
    V_futuro_singular -> 'observará' | 'comerá' | 'adorará'
    V_futuro_plural -> 'observarán' | 'comerán' | 'adorarán'
    P -> 'con'
  """
)

# Create a parser with the defined grammar
parser = nltk.ChartParser(gramatica)

# Input sentence
# oracion = "los hermosos tigres observaron el grande caballo"

oracion = "¿ los hermosos tigres observaron el grande caballo ?"

# Remove punctuation and tokenize the sentence
translator = str.maketrans("", "", string.punctuation)
oracion_sin_puntuacion = oracion.translate(translator)
tokens = oracion_sin_puntuacion.split()

try:
  # Try to parse the sentence with the defined grammar
  arbol_parseo = list(parser.parse(tokens))

  # Show the grammar productions
  print("\nProducciones de la gramática:")
  for produccion in gramatica.productions():
    print(produccion)

  # Show the parse tree
  print("\nÁrbol de parseo:")
  arbol_parseo[0].pretty_print()

  # Find the verb in the parse tree and determine its tense and number
  tiempo_verbo = ""
  numero_verbo = ""
  for subarbol in arbol_parseo[0].subtrees():
    if subarbol.label() == "V_presente_singular":
      tiempo_verbo = "presente"
      numero_verbo = "singular"
    elif subarbol.label() == "V_presente_plural":
      tiempo_verbo = "presente"
      numero_verbo = "plural"
    elif subarbol.label() == "V_pasado_singular":
      tiempo_verbo = "pasado"
      numero_verbo = "singular"
    elif subarbol.label() == "V_pasado_plural":
      tiempo_verbo = "pasado"
      numero_verbo = "plural"
    elif subarbol.label() == "V_futuro_singular":
      tiempo_verbo = "futuro"
      numero_verbo = "singular"
    elif subarbol.label() == "V_futuro_plural":
      tiempo_verbo = "futuro"
      numero_verbo = "plural"

  if tiempo_verbo:
    print(
      f"\nEl verbo utilizado está en tiempo {tiempo_verbo} y es {numero_verbo}."
    )
  else:
    print("\nNo se pudo determinar el tiempo verbal y número del verbo.")

  print("\nLa oración cumple con las reglas gramaticales.")

except IndexError:
  print("\nLa oración no cumple con las reglas gramaticales.")



