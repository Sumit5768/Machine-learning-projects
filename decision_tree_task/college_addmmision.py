import sys
import matplotlib
matplotlib.use('Agg')

import pandas as p
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


df = p.read_csv("college_admission.csv")

d = {"High":2,"Medium":1,"Low":0}
df["12th_Percentage"] = df["12th_Percentage"].map(d)
d = {"Pass":1,"Fail":0}
df["Entrance_Exam_Result"] = df["Entrance_Exam_Result"].map(d)
d = {"General":1,"reserved":0}
df["Category"] = df["Category"].map(d)
d = {"Yes":1,"No":0}
df["Admission_Granted"] = df["Admission_Granted"].map(d)

features = ["12th_Percentage","Entrance_Exam_Result","Category"]
 
x = df[features]
y = df["Admission_Granted"]

admission = DecisionTreeClassifier()
admission.fit(x,y)

AdmissionResult = admission.predict([[2,1,1]])
print(AdmissionResult)