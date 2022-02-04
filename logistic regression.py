import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/TEMP/Documents/Personal/pythonProject/logistic_dataset.csv")
print(df.head())

plt.scatter(df['age'], df['bought insurance'])
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
x_train, x_test, y_train, y_test = train_test_split(df[['age']],df['bought insurance'],train_size=0.8)
print(x_test)

model = LogisticRegression()

model.fit(x_train, y_train)

y_predicted = model.predict(x_test)
model.predict_proba(x_test)
print(model.predict_proba(x_test))


model.score(x_test, y_test)
print(model.score(x_test, y_test))
print(y_predicted)



print(model.coef_)

print(model.intercept_)

import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def prediction_function(age):
    z = 0.042 * age - 1.53 
    y = sigmoid(z)
    return y

age = 20
print(prediction_function(age))

age = 60
print(prediction_function(age))













