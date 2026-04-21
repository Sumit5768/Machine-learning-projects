import sys
import matplotlib
matplotlib.use('Agg')

import pandas as p
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = p.read_csv("student_pass_fail.csv")

d = {"Low":0,"Medium":1,"High":2}
df["Attendance"]  = df["Attendance"].map(d)
df["StudyHours"]  = df["StudyHours"].map(d)

d = {"Fail":0,"Pass":1}
df["InternalTest"]  = df["InternalTest"].map(d)

df["FinalResult"]  = df["FinalResult"].map(d)

features = ["Attendance","StudyHours","InternalTest"]

x = df[ features]
y = df["FinalResult"]

dtree = DecisionTreeClassifier()
dtree.fit(x,y)

result = dtree.predict([[0,1,1]])
print(result)