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
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Prediction
input_to_predict = torch.tensor([1, 0, 1, 0, 0, 0 , 0, 1], dtype=torch.float32)
output = model(input_to_predict)
predicted_class = torch.argmax(output).item()
predicted_label = target_encoder.inverse_transform([predicted_class])[0]
print("Predicted output:", predicted_label)



# print(binary_data)
