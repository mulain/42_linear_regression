import numpy as np
import pandas as pd
import plotly.express as px

df = pd.read_csv('data.csv')


fig = px.scatter(df, x='km', y='price', title='Mileage vs Price')

# Show the plot
fig.show()

# def estimate_price(mileage, theta0, theta1):
#     return theta0 + theta1 * mileage

# def cost_function(mileages, prices, theta0, theta1):
#     m = len(mileages)
#     total_error = 0.0
    
#     for i in range(m):
#         total_error += (estimate_price(mileages[i], theta0, theta1) - prices[i]) ** 2
#     return total_error / (2 * m)

# def gradient_descent(mileages, prices, theta0, theta1, learning_rate, iterations):
#     m = len(mileages)
#     for _ in range(iterations):
#         sum_errors_theta0 = 0.0
#         sum_errors_theta1 = 0.0
#         for i in range(m):
#             error = estimate_price(theta0, theta1, mileages[i]) - prices[i]
#             sum_errors_theta0 += error
#             sum_errors_theta1 += error * mileages[i]
#         theta0 -= (learning_rate  m) * sum_errors_theta0
#         theta1 -= (learning_rate  m) * sum_errors_theta1
#     return theta0, theta1

# def main():
#     df = pd.read_csv('data.csv')
#     mileages = df['km'].values
#     prices = df['price'].values
#     learning_rate = 0.1
#     iterations = 1000

#     theta0, theta1 = gradient_descent(mileages, prices, 0.0, 0.0, learning_rate, iterations)

# def train(mileages, prices):
#     theta0 = 0.0
#     theta1 = 0.0

#     learning_rate = 0.1
#     iterations = 1000

#     theta0, theta1 = gradient_descent(mileages, prices, theta0, theta1, learning_rate, iterations)

#     return theta0, theta1

# def gradient_descent(mileages, prices, theta0, theta1, learning_rate, iterations):
#     m = len(mileages)
#     for _ in range(iterations):
#         theta0, theta1 = step_gradient(mileages, prices, theta0, theta1, learning_rate)

#     return theta0, theta1



