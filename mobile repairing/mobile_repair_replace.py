import sys
import matplotlib
matplotlib.use('Agg')

import pandas as p
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


df = p.read_csv("mobile_repair_replace.csv")

d = {"New":1,"Old":0}
df["Phone_Age"] = df["Phone_Age"].map(d)
d = {"Low":0,"High":1}
df["Repair_Cost"] = df["Repair_Cost"].map(d)
d = {"Premium":1,"Normal":0}
df["Brand_Type"] = df["Brand_Type"].map(d)
d = {"Replace":1,"Repair":0}
df["Decision"] = df["Decision"].map(d)

features = ["Phone_Age","Repair_Cost","Brand_Type"]
 
x = df[features]
y = df["Decision"]

mobile = DecisionTreeClassifier()
mobile.fit(x,y)

MobileResult = mobile.predict([[2,1,1]])
print(MobileResult)