import numpy as np
import pandas as pd

def main():
    theta0 = 0.0
    theta1 = 0.0
    
    dataFrame = pd.read_csv('data.csv')
    mileages = dataFrame['km'].values
    prices = dataFrame['price'].values

def estimatePrice(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def costFunction()