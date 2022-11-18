import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
path = "./assets/Age_And_Time.csv"
ageAndTimeDf = pd.read_csv(path)
path = "./assets/countries.csv"
countriesDf = pd.read_csv(path)
path = "./assets/heights.csv"
heightsDf = pd.read_csv(path)

#1.a

heights = heightsDf["HEIGHT"]
weights = heightsDf["WEIGHT"]
plt.scatter(heights, weights)
# plt.show()
#2.b
items = [(heights[i], weights[i]) for i in range(len(heights))]
sortedHeights = sorted(items, key=lambda item: item[0])
lindex = (len(sortedHeights)//6)
rindex = (len(sortedHeights)//6)*5
Xl = sortedHeights[lindex][0]
Xr = sortedHeights[rindex][0]
Yl=np.median([sortedHeights[i][1] for i in range(len(sortedHeights)//3)])
Yr=np.median([sortedHeights[i][1] for i in range((len(sortedHeights)//3)*2, len(sortedHeights))])
Brl = (Yr-Yl)/(Xr-Xl)
allRs= [sortedHeights[i][1]-Brl*sortedHeights[i][0] for i in range(len(sortedHeights))]
Arl = np.median(allRs)

x = np.linspace(50,250,100)
y = x*Brl+Arl
plt.plot(x, y, '-r', label='resistant')

temp = [[items[i][0], items[i][1]]for i in range(len(items))]
Bls=(np.cov(weights, heights, ddof=0)[0][1])/np.var(heights)

Als = np.mean(weights)-Bls*np.mean(heights)
x = np.linspace(50,250,100)
y = x*Bls+Als
plt.plot(x, y, '-b', label='sum of squares')


plt.show()
r= np.cov(heights,weights, ddof=0)[0][1]/(np.std(heights)*np.std(weights))
print(Bls)
print(r)
print((r**2)*100)