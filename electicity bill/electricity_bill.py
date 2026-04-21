import pandas as pd
from sklearn.linear_model import LinearRegression

dt = pd.read_csv("electricity.csv")

X = dt[["AC_Usage_Hours", "Number_of_Fans", "Units_Consumed"]]
y = dt["Monthly_Bill"]

bill_model = LinearRegression()
bill_model.fit(X, y)

predicted_bill = bill_model.predict([[4, 3, 200]])

print(predicted_bill)
