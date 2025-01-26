import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import optuna

# Load and prepare the dataset
def prepare_data():
    # Load Iris dataset as an example
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Objective function for Optuna
def objective(trial):
    # Define the hyperparameters to tune
    C = trial.suggest_loguniform('C', 1e-5, 1e5)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1e4)
    
    # Create and train the model
    model = SVC(C=C, kernel=kernel, gamma=gamma)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    score = model.score(X_test, y_test)
    return score

def main():
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    # Print the best parameters and score
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Train the final model with best parameters
    best_params = study.best_params
    final_model = SVC(**best_params)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Fit the final model
    final_model.fit(X_train, y_train)
    
    # Save the model
    import joblib
    joblib.dump(final_model, 'best_model.joblib')
    
    return final_model, best_params

if __name__ == "__main__":
    main()