import sys
import matplotlib
matplotlib.use('Agg')

import pandas as p
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = p.read_csv("loan_approval.csv")

d = {"High":2,"Medium":1,"Low":0}
df["Income_Level"] = df["Income_Level"].map(d)
d = {"Good":1,"Bad":0}
df["CIBIL_Score"] = df["CIBIL_Score"].map(d)
d = {"Govt":1,"Private":0}
df["Job_Type"] = df["Job_Type"].map(d)
d = {"Yes":1,"No":0}
df["Loan_Approved"] = df["Loan_Approved"].map(d)

features = ["Income_Level","CIBIL_Score","Job_Type"]

x = df[features]
y = df["Loan_Approved"]

loan = DecisionTreeClassifier()
loan.fit(x,y)


LoanResult = loan.predict([[2,1,1]])
print(LoanResult)