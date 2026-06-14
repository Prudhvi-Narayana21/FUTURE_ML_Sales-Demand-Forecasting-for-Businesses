import pandas as pd
from sklearn.linear_model import LinearRegression

print("Sales Demand Forecasting Project")

data = {
    "Month": [1, 2, 3, 4, 5],
    "Sales": [100, 120, 140, 160, 180]
}

df = pd.DataFrame(data)

X = df[["Month"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[6]])

print("Predicted Sales for Month 6:", prediction[0])
