import pandas as p
from sklearn.linear_model import LinearRegression

dt = p.read_csv("youtube_views.csv")

x = dt[["Video_Length","Thumbnail_Quality","Posting_Time"]]
y = dt[["Views_24h"]]

view = LinearRegression()
view.fit(x,y)

view_24 = view.predict([[7,6,14]])
print("views in 24 hours in youtube :",view_24)