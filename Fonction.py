from dataclasses import dataclass
import pandas as pd

data = pd.read_csv('Student_Performance.csv')

def empirical_risk(data,func,y1=0,y2=1):
    """
    :param data: Dataset of your choice
    :param func: Linear regression or predictor we want to calculate the error for
    :param y1: Dataset y parameter
    :param y2: Dataset yn parameter
    :return: Total error

    """
    total_error = 0
    for i in  range (len(data)):
        yn = data.iloc[i].iloc[y1]
        yr = func(data.iloc[i].iloc[y2])
        total_error += (yn - yr) ** 2
    total_error = total_error/len(data)
    return total_error

# Empirical Risk Minimization
risk_min = []  # Store all (p1, p2, risk) values
best_params = None
best_risk = float('inf')  # Initialize with high risk


for i in range(1000):
    p1 = i * 0.01  # Testing different values of p1
    for j in range(1000):
        p2  = j * 0.01  # Testing different values of p2
        f = lambda x: x * p1 + p2  # Hypothesis function (linear predictor)
        emp = empirical_risk(data, f, y1=1, y2=5)  # Compute risk
        risk_min.append([p1, p2, emp])
        # Update best parameters if lower risk is found
        if emp < best_risk:
            best_risk = emp
            best_params = (p1, p2)


print(f"Best parameters: p1 = {best_params[0]}, p2 = {best_params[1]}, Risk = {best_risk}")




