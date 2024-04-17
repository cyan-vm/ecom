import torch
from nn import model, output, target_encoder 

test = torch.tensor([1, 0, 1, 0, 0, 0 , 0, 1], dtype=torch.float32)

model(test)

predicted_test = torch.argmax(output).item()
predicted_label_test = target_encoder.inverse_transform([predicted_test])[0]
print("Predicted output: ", predicted_label_test)

# # Define the input size, hidden size, and output size
# input_size = len(data.columns[:-1])
# hidden_size = 64
# output_size = len(data['depresion'].unique())

# # Instantiate the model
# model = NeuralNetwork(input_size, hidden_size, output_size)

# # Load the saved model state dictionary
# model.load_state_dict(torch.load("best_model.pth"))

# # Make predictions using the loaded model
# input_to_predict = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.float32)
# output = model(input_to_predict)
# predicted_class = torch.argmax(output).item()
# print("Predicted class:", predicted_class)

