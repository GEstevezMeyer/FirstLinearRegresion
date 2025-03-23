import pandas as pd
import numpy as np


data = pd.read_csv('Student_Performance.csv')

def empirical_risk(data, func, x_col, y_col):
    total_error = 0
    n = len(data)
    for i in range(n):
        x = data.iloc[i, x_col]
        y = data.iloc[i, y_col]
        y_pred = func(x)
        total_error += (y - y_pred) ** 2
    return total_error / n


def gradient_descent(data, p1, p2, L, epochs, x_col, y_col):
    n = len(data)

    for _ in range(epochs):
        p1_gradient = 0
        p2_gradient = 0

        for i in range(n):
            x = data.iloc[i, x_col]
            y = data.iloc[i, y_col]

            error = y - (p1 * x + p2)
            p1_gradient += (-2 / n) * x * error
            p2_gradient += (-2 / n) * error

        p1 -= L * p1_gradient
        p2 -= L * p2_gradient

    return p1, p2
#(ERM)
risk_min = []
best_params = None
best_risk = float('inf')

for i in np.linspace(-10, 10, 100):
    p1 = i
    for j in np.linspace(-50, 50, 100):
        p2 = j
        f = lambda x: p1 * x + p2
        emp = empirical_risk(data, f, x_col=1, y_col=5)
        risk_min.append([p1, p2, emp])

        if emp < best_risk:
            best_risk = emp
            best_params = (p1, p2)

print(f"ERM: p1 = {best_params[0]:.4f}, p2 = {best_params[1]:.4f}, Risk = {best_risk:.4f}")



