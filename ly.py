import torch 
from trial import model, load_model, input_size, best_hidden_size, output_size, model_filepath, X, y, target_encoder
# import torch.nn as nn
# import torch.optim as optim

model = load_model(input_size, best_hidden_size, output_size, model_filepath)

print("Model loaded successfully.")

# Prediction
correct_predictions = 0
wrong_predictions = 0

for i in range(len(X)):
    input_to_predict = X[i]
    output = model(input_to_predict)
    predicted_class = torch.argmax(output).item()
    true_class = y[i].item()
    if predicted_class == true_class:
        correct_predictions += 1
    else:
        wrong_predictions += 1

print("Number of correct predictions:", correct_predictions)
print("Number of wrong predictions:", wrong_predictions)

print(f"Accuracy : {correct_predictions / len(X) * 100}%")


# Prediction
input_to_predict = torch.tensor([0, 0, 1, 1, 0, 0, 0, 0], dtype=torch.float32)
output = model(input_to_predict)
predicted_class = torch.argmax(output).item()
predicted_label = target_encoder.inverse_transform([predicted_class])[0]
print("Predicted output:", predicted_label)



# [1, 0, 0, 0, 0, 0, 0, 0]


# input_to_predict = torch.tensor([1, 0, 1, 0, 0, 0 , 0, 1], dtype=torch.float32)
# output = model(input_to_predict)
# predicted_class = torch.argmax(output).item()
# predicted_label = target_encoder.inverse_transform([predicted_class])[0]
# print("Predicted output:", predicted_label)

# print(LabelEncoder())



