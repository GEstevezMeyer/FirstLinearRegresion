import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Student_Performance.csv")
plt.scatter(data.Previous_Scores,data.Performance_Index)
plt.plot(list(range(20,100)),[(x*1.1111-21.7172) for x in list(range(20,100))])
plt.show()
