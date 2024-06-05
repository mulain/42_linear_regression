import numpy as np
import pandas as pd
import plotly.express as px

# normalization

def min_max_normalize(arr, feature_range=(0, 1)):
    """
    Perform Min-Max normalization on a NumPy array.

    Parameters:
    - arr: NumPy array to be normalized.
    - feature_range: Tuple specifying the desired range of the normalized values (default: (0, 1)).

    Returns:
    - Normalized NumPy array.
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    min_feature, max_feature = feature_range

    # Normalize the array
    normalized_arr = (arr - min_val) / (max_val - min_val) * (max_feature - min_feature) + min_feature

    return normalized_arr


def estimate_dependent(theta0, theta1, X):
    return theta0 + theta1 * X

def gradient_descent(X, y, theta0, theta1, learning_rate, iterations):
    m = len(X)
    
    for i in range(iterations):
        predictions = estimate_dependent(theta0, theta1, X)
        error = predictions - y
        avg_error0 = np.sum(error) / m
        avg_error1 = np.sum(error * X) / m
        
        temp0 = theta0 - learning_rate * avg_error0
        temp1 = theta1 - learning_rate * avg_error1
        
        theta0, theta1 = temp0, temp1

        print(f"Iteration {i}: Average error: {avg_error0}, Average error * X: {avg_error1}")
    
    return theta0, theta1

# Load dataset
df = pd.read_csv('data.csv')

# Check for NaN or infinite values
if df.isnull().any() or not np.isfinite(df).all():
    raise ValueError("Data contains NaN or infinite values.")

X = df['km'].values
y = df['price'].values

# Initial parameters and hyperparameters
learning_rate = 0.0000001
iterations = 100
theta0 = 0.0
theta1 = 0.0

# Plot the data
fig = px.scatter(df, x='km', y='price', title='Mileage vs Price')
fig.show()

# Train the model
theta0, theta1 = gradient_descent(min_max_normalize(X), y, theta0, theta1, learning_rate, iterations)

print(f"Trained parameters: theta0 = {theta0}, theta1 = {theta1}")

# Save the parameters
with open('model_parameters.txt', 'w') as file:
    file.write(f"{theta0}, {theta1}")

"""
Intercept(θ0)=6719.571336173393
Coefficient(θ1)=-0.0214480462745097
"""
