import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(X, y, model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(input_size, hidden_size, output_size, filepath):
    model = NeuralNetwork(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(filepath))
    return model

data = pd.read_excel("~/Downloads/salidas.xlsx")

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

input_size = len(data.columns[:-1])
hidden_size = 93
output_size = len(data['depresion'].unique())

# Instantiate the model or load from a saved file
model_filepath = 'model.pth'
try:
    model = load_model(input_size, hidden_size, output_size, model_filepath)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Training a new model...")
    model = NeuralNetwork(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.009827461337674192)
    epochs = 100
    train_model(X, y, model, criterion, optimizer, epochs)
    save_model(model, model_filepath)
    print("Model trained and saved.")

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
input_to_predict = torch.tensor([1, 0, 0, 1, 1, 0, 0, 1], dtype=torch.float32)
output = model(input_to_predict)
predicted_class = torch.argmax(output).item()
predicted_label = target_encoder.inverse_transform([predicted_class])[0]
print("Predicted output:", predicted_label)

