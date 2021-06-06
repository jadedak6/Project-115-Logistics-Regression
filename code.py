from google.colab import files
data_to_load = files.upload()


import pandas as pd
import plotly.express as px

df= pd.read_csv("as.csv")

velocity_list = df["Velocity"].tolist()
escaped_list = df["Escaped"].tolist()

fig = px.scatter(x=velocity_list, y=escaped_list)
fig.show()


import numpy as np
temperature_array = np.array(velocity_list)
melted_array = np.array(escaped_list)

m, c = np.polyfit(temperature_array, melted_array, 1)

y = []
for x in temperature_array:
  y_value = m*x + c
  y.append(y_value)

  fig = px.scatter(x=temperature_array, y=melted_array)
  fig.update_layout(shapes=[
      dict(
          type= 'line',
          y0 = min(y), y1 = max(y),
          x0 = min(temperature_array), x1 = max(temperature_array)
      )
  ])
  fig.show()
  
  import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.reshape(velocity_list, (len(velocity_list), 1))
Y = np.reshape(escaped_list, (len(escaped_list), 1))

lr = LogisticRegression()
lr.fit(X, Y)

plt.figure()
plt.scatter(X.ravel(), Y, color='black', zorder=20)

def model(x):
  return 1 / (1 + np.exp(-x))

  x_test = np.linspace(0, 5000, 10000)
  melting_changes = model(X_test * lr.coef_ + lr.intercept_).ravel()

  plt.plot(X_test, melting_changes, color='red', linewidht=3)
  plt.axhline(y=0, color='k', linestyle='-')
  plt.axhline(y=1, color='k', linestyle='-')
  plt.axhline(y=0.5, color='b', linestyle='--')

  plt.axhline(yx=X_test[6843], color='b', linestyle='--')
  
  plt.ylabel('y')
  plt.xlabel('X')
  plt.xlim(3400, 3450)
  print(X_test[6843])
  
  temp = float(input("Enter the temperature here:-"))
chances = model(temp * lr.coef_ + lr.intercept_).ravel()[0]
if chances <= 0.01:
  print("Tungsten will not be melted")
elif chances <= 1:
  print("Tungsten will be melted")
elif chances <= 1:
  print("Tungsten will be melted")
  
