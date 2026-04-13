import pandas as p
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

dt = p.read_csv("9.csv")

x = dt[["Classroom_Temperature","Noise_Level","Lecture_Duration"]]
y = dt["Attention"]

a = LinearRegression()
a.fit(x,y)

predicted_attention = a.predict([[25,33,25]])
print(predicted_attention)



if predicted_attention > 60:
    print("Decision: Continue Class")
else:
    print("Decision: Take Break / Activity")
