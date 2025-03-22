import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Student_Performance.csv")
plt.scatter(data.Previous_Scores, data.Performance_Index)
plt.plot(list(range(0,100)),[x*1.16+1.98 for x in list(range(0,100))])
plt.show()
