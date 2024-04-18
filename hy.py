import pandas as pd

# Read the data
data = pd.read_excel("~/Downloads/salidas.xlsx")

# Define a mapping for 'sexo' column
sexo_mapping = {'Hombre': 0, 'Mujer': 1}

# Define a mapping for 'depresion' column
depresion_mapping = {'No': 0, 'Si': 1, 'Propenso': 2}

# Define a mapping for other columns
other_columns_mapping = {'No': 0, 'Si': 1}

# Convert 'sexo' column
data['Sexo'] = data['Sexo'].replace(sexo_mapping)

# Convert 'depresion' column
data['depresion'] = data['depresion'].replace(depresion_mapping)

# Convert other specified columns
columns_to_convert = ['Miedo al futuro', 'Perdida de interes', 'DistraccionenClase', 'ProblemasRP', 'CambiosFisicosN', 'Acoso', 'CambioApetito']
data[columns_to_convert] = data[columns_to_convert].replace(other_columns_mapping)

# Convert object type columns to numeric
object_columns = ['Sexo', 'Miedo al futuro', 'Perdida de interes', 'DistraccionenClase', 'ProblemasRP', 'CambiosFisicosN', 'Acoso', 'CambioApetito', 'depresion']
data[object_columns] = data[object_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
data.dropna(inplace=True)

# Split the data into input features (X) and target variable (y)
X = data[['Sexo', 'Miedo al futuro', 'Perdida de interes', 'DistraccionenClase', 'ProblemasRP', 'CambiosFisicosN', 'Acoso', 'CambioApetito']].values
y = data['depresion'].values

# Convert to PyTorch tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.int64)

# Define the neural network architecture
# Adjusted neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 32)  # Increased neurons in the first layer
        self.fc2 = nn.Linear(32, 64)  # Added an extra hidden layer
        self.fc3 = nn.Linear(64, 3)   # 3 output classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Applying oversampling to balance the dataset
from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X, y)

# Convert to PyTorch tensors
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.int64)

# Define the model, loss function, and optimizer
model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    model.eval()
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f'Accuracy: {accuracy:.2f}')

# Test prediction
input_features = torch.tensor([0, 0, 1, 0, 0, 0, 0, 1], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
prediction = model(input_features)
predicted_class = torch.argmax(prediction, dim=1).item()
print(f'Predicted class: {predicted_class}')

# Display the transformed data
# print(data.head())

# Initialize variables to count correct and wrong predictions
correct_predictions = 0
wrong_predictions = 0

# Loop through each row of the data
for i in range(len(data)):
    # Extract input features and target label for the current row
    input_features = torch.tensor(data.iloc[i, :-1].values, dtype=torch.float32)
    target_label = torch.tensor(data.iloc[i, -1], dtype=torch.int64)
    
    # Make prediction
    with torch.no_grad():
        model.eval()
        output = model(input_features)
        predicted_class = torch.argmax(output).item()
        # print(predicted_class, target_label)
    
    # Compare predicted class with actual label
    if predicted_class == target_label:
        correct_predictions += 1
    else:
        wrong_predictions += 1

# Calculate total accuracy
total_predictions = correct_predictions + wrong_predictions
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

# Print results
print(f"Number of correct predictions: {correct_predictions}")
print(f"Number of wrong predictions: {wrong_predictions}")
print(f"Total accuracy: {accuracy:.2f}")


