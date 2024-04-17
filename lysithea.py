import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

# Read the data
data = pd.read_excel("~/Downloads/salidas.xlsx")

# Define function to convert categorical features to binary encoding
def convert_to_binary(df):
    binary_data = df.copy()
    for column in binary_data.columns[:-1]:  # Exclude the target column
        binary_data[column] = binary_data[column].apply(lambda x: 0 if x == 'No' else 1)
    return binary_data

# Convert categorical features to binary encoding
binary_data = convert_to_binary(data)

# target_encoder = LabelEncoder()
# binary_data['depresion'] = target_encoder.fit_transform(binary_data['depresion'])

# Convert categorical variables to numerical using LabelEncoder
label_encoders = {}
for column in data.columns[:-1]:  # Exclude the target column
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Convert target variable to numerical
target_encoder = LabelEncoder()
data['depresion'] = target_encoder.fit_transform(data['depresion'])

# Define input and target variables
X = data[data.columns[:-1]].values
y = data['depresion'].values

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the input size, hidden size, and output size
input_size = len(data.columns[:-1])
hidden_size = 64
output_size = len(data['depresion'].unique())

# Instantiate the model
model = NeuralNetwork(input_size, hidden_size, output_size)

import torch.optim as optim

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Extracting rows of input features
data_input = binary_data.iloc[:4, :-1].values

# Converting to PyTorch tensor
data_input = torch.tensor(data_input, dtype=torch.float32)

# Prediction
input_to_predict = data_input[0].clone().detach()
output = model(input_to_predict)
predicted_class = torch.argmax(output).item()
predicted_label = target_encoder.inverse_transform([predicted_class])[0]
print("Predicted output:", predicted_label)


# Initialize counters for correct and incorrect predictions
correct_predictions = 0
incorrect_predictions = 0

# Loop through all rows in the data
for i in range(len(binary_data)):
    # Extract input features for the current row
    data_input = torch.tensor(binary_data.iloc[i, :-1].values, dtype=torch.float32)

    # Make prediction
    output = model(data_input)
    predicted_class = torch.argmax(output).item()
    predicted_label = target_encoder.inverse_transform([predicted_class])[0]

    # Compare prediction to actual value
    actual_label = binary_data.iloc[i, -1]
    if predicted_label == actual_label:
        correct_predictions += 1
    else:
        incorrect_predictions += 1

# Print the results
print(f'Correct predictions: {correct_predictions}')
# print(f'Incorrect predictions: {incorrect_predictions}')


total_predictions = len(X)
percentage_correct = (correct_predictions / total_predictions) * 100
percentage_incorrect = (incorrect_predictions / total_predictions) * 100

# print("accurary", percentage_correct, '%')
print(f'accurary : {percentage_correct}%')
# print("Percentage of incorrect predictions:", percentage_incorrect)

# print(binary_data)
