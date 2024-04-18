import pandas as pd

# Read the data
data = pd.read_excel("~/Downloads/salidas.xlsx")

# Define the mapping dictionary for binary columns
binary_mapping = {'No': 0, 'Si': 1}

# Replace values in the columns with binary values
binary_columns = ['Miedo al futuro', 'Perdida de interes', 'DistraccionenClase', 
                  'ProblemasRP', 'CambiosFisicosN', 'Acoso', 'CambioApetito']

for column in binary_columns:
    data[column] = data[column].replace(binary_mapping)

# Define the mapping dictionary for 'depresion' column
depresion_mapping = {'No': 0, 'Si': 1, 'Propenso': 2}

# Replace values in the 'depresion' column
data['depresion'] = data['depresion'].replace(depresion_mapping)

# Define the mapping dictionary for 'Sexo' column
sexo_mapping = {'Hombre': 0, 'Mujer': 1}

# Replace values in the 'Sexo' column
data['Sexo'] = data['Sexo'].replace(sexo_mapping)

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

# Convert data to numpy arrays
X = data.drop('depresion', axis=1).values.astype(np.float32)  # Convert to float32 explicitly
y = data['depresion'].values.astype(np.int64)  # Convert to int64 explicitly

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define the Neural Network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Initialize the model
input_size = X_train.shape[1]
hidden_size = 16
output_size = 3  # Adjusted output size based on the number of unique values in 'depresion' column
model = NeuralNetwork(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

    # Convert the binary array to tensor
    input_tensor = torch.tensor([0, 0, 1, 0, 0, 0, 0, 1], dtype=torch.float32)
    # Make prediction
    prediction = torch.argmax(model(input_tensor))
    print("Prediction for input array [0, 0, 1, 0, 0, 0, 0, 1]:", prediction.item())
