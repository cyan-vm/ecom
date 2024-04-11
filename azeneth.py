import nltk
import string

# Defining the grammar with information about the verb tense and number
gramatica = nltk.CFG.fromstring(
    """
O -> SN SV
SN -> Det N | Det Adj N | Det Adj Adj N | SN SP
SV -> V | V SN | V SN SN | V SP | V SN SP | V SP SN
SP -> Prep SN | Prep NProp
Det -> 'el' | 'la' | 'los' | 'las' | 'un' | 'una' | 'unos' | 'unas' | 'este' | 'esta' | 'estos' | 'estas' | 'ese' | 'esa' | 'esos' | 'esas' | 'mi' | 'tu' | 'su' | 'nuestro' | 'vuestro' | 'sus' | 'algunos' | 'algunas' | 'todas' | 'todos' |'solo' | 'Solo' | 'El' | 'La' | 'Los' | 'Las' | 'Un' | 'Una' | 'Unos' | 'Unas' | 'Este' | 'Esta' | 'Estos' | 'Estas' | 'Ese' | 'Esa' | 'Esos' | 'Esas' | 'Mi' | 'Tu' | 'Su' | 'Nuestro' | 'Vuestro' | 'Sus' | 'Algunos' | 'Algunas' | 'Todas' | 'Todos'
N -> 'joven' | 'chico' | 'chica' | 'mandarinas' | 'lapiz' | 'perro' | 'gato' | 'casa' | 'temprano' | 'mesa' | 'silla' | 'niño' | 'niña' | 'mujer' | 'hombre' | 'computadora' | 'teléfono' | 'libro' | 'auto' | 'bicicleta' | 'ciudad' | 'país' | 'amigo' | 'amiga' | 'familia' | 'comida' | 'estudia' | 'mañanas' | 'trabaja' | 'tardes' | 'es' | 'apellido' | 'muy' | 'común' | 'región'
NProp -> 'Azeneth' | 'Iveth' | 'Rosas' | 'María' | 'José' | 'Juan' | 'Ana' | 'Luis' | 'Carlos' | 'Laura' | 'David' | 'Carmen' | 'Manuel' | 'Rosa' | 'Pedro' |' Patricia' | 'Antonio' | 'Isabel' | 'García' | 'Martínez' | 'Pérez' | 'González'
Adj -> 'hermoso' | 'pequeña' | 'plata' | 'listo' | 'bueno' | 'bien' | 'fiel' | 'fácil' | 'frio' | 'feo' | 'claro' | 'barato' | 'débil' | 'divertido' | 'frito' | 'enfermo' | 'libre' | 'limpio' | 'sucio' | 'pequeño' | 'lleno' | 'gastado' | 'largo' | 'malo' | 'feliz' | 'atrevido' | 'simpatico' | 'injusto' | 'paciente' | 'aburrido' | 'cuidado' | 'humano' | 'elegante' | 'cerrado' | 'egoísta' | 'alegre' | 'triste' | 'tranquilo' | 'nervioso'
V -> 'comer' | 'salta' | 'pela' | 'persigue' | 'ser' | 'estar' | 'haber' | 'tener' | 'ir' | 'venir' | 'poder' | 'querer' | 'saber' | 'conocer' | 'decir' | 'ver' | 'oir' | 'escuchar' | 'poner' | 'sacar' | 'salir' | 'salíamos' | 'entrar' | 'vivir' | 'trabajar' | 'estudiar' | 'correr' | 'nadar' | 'bailar' | 'cantar' | 'gritar' | 'jugar' | 'dormir'
Prep -> 'a' | 'ante' | 'bajo' | 'jugosas' | | 'contra' | 'de' | 'desde' | 'en' | 'entre' | 'hacia' | 'hasta' | 'para' | 'por' | 'según' | 'durante' | 'mediante' | 'sobre' | 'sin' | 'tras' | 'pro' | 'con' | 'durante' | 'tras' | 'mediante'
    """
)

# Create a parser with the defined grammar
parser = nltk.ChartParser(gramatica)

# Input sentence
oracion = "Todas las mañanas salíamos bien temprano solo para comer mandarinas jugosas"

isInterrogation = False

# Check if the sentence ends with a question mark
if oracion.startswith("¿") and oracion.endswith("?"):
    # Remove the question mark
    oracion = oracion[1:]
    oracion = oracion[:-1]
    isInterrogation = True

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

if isInterrogation:
    print("La oracion es una pregunta")
