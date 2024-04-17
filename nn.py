import optuna

def objective(trial):
    # Define hyperparameters to search
    hidden_size = trial.suggest_int('hidden_size', 32, 128)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    # Instantiate the model with the suggested hyperparameters
    model = NeuralNetwork(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    # Evaluate the model on validation data (not shown in your code)
    # For simplicity, you can directly evaluate on the training data here
    
    correct_predictions = 0
    for i in range(len(X)):
        input_to_predict = X[i]
        output = model(input_to_predict)
        predicted_class = torch.argmax(output).item()
        true_class = y[i].item()
        if predicted_class == true_class:
            correct_predictions += 1
    accuracy = correct_predictions / len(X)
    
    # Return negative accuracy because Optuna tries to minimize the objective
    return -accuracy

# Perform hyperparameter optimization with Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params

# Train the model with the best hyperparameters
best_hidden_size = best_params['hidden_size']
best_learning_rate = best_params['learning_rate']
best_model = NeuralNetwork(input_size, best_hidden_size, output_size)
best_optimizer = optim.Adam(best_model.parameters(), lr=best_learning_rate)
for epoch in range(epochs):
    best_optimizer.zero_grad()
    outputs = best_model(X)
    loss = criterion(outputs, y)
    loss.backward()
    best_optimizer.step()

# Evaluate the best model
correct_predictions = 0
for i in range(len(X)):
    input_to_predict = X[i]
    output = best_model(input_to_predict)
    predicted_class = torch.argmax(output).item()
    true_class = y[i].item()
    if predicted_class == true_class:
        correct_predictions += 1
accuracy = correct_predictions / len(X)

print("Best hyperparameters:", best_params)
print("Accuracy of the best model:", accuracy)


