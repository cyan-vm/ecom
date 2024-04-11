import re

def check_question_marks(sentence):
    # Regular expression pattern to find opening and closing question marks with a word in between
    pattern = r'¿(\w+)\?'

    # Search for the pattern in the sentence
    match = re.search(pattern, sentence)

    # If a match is found, return True, otherwise False
    if match:
        return True
    else:
        return False

# Example sentence
sentence = '¿los hermosos tigres observaron el grande caballo?'

# Check if the sentence contains a word wrapped between opening and closing question marks
result = check_question_marks(sentence)

# Print the result
print(result)
