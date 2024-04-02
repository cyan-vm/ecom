import nltk
from nltk import RegexpParser
from nltk.tokenize import word_tokenize

# Definir un diccionario manualmente
dictionary = {
    "comer": ["yo como", "tú comes", "él/ella come", "nosotros/as comemos", "vosotros/as coméis", "ellos/ellas comen"]
}

# Oración de entrada
sentence = "Yo como una manzana verde."

# Tokenización de la oración
tokens = word_tokenize(sentence)

# Etiquetado gramatical
tagged = nltk.pos_tag(tokens)

# Reglas de la gramática
grammar = r"""
    NP: {<DT>?<JJ>*<NN>} # Chunk determiner/adj/noun
    VP: {<PRP>|<VB.*><NP|PP>*} # Chunk verbs and their arguments
"""

# Crear el analizador de sintaxis
chunk_parser = RegexpParser(grammar)

# Parsear la oración
parsed_sentence = chunk_parser.parse(tagged)

# Imprimir el árbol
parsed_sentence.pretty_print()

# Identificar conjugación del verbo
verb = ""
for word, pos in tagged:
    if pos.startswith('VB'):
        verb = word
        break

# Buscar en el diccionario manual
pronoun = ""
if verb in dictionary:
    for entry in dictionary[verb]:
        pronoun, conjugated_verb = entry.split(" ", 1)
        if pronoun.lower() == "yo" and conjugated_verb == verb:
            break

# Imprimir resultados
if pronoun:
    print("La oración está correctamente conjugada con el pronombre:", pronoun)
else:
    print("La oración no está en el diccionario o no está correctamente conjugada.")
