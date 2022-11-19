import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

path = "./assets/Age_And_Time.csv"
ageAndTimeDf = pd.read_csv(path)
path = "./assets/countries.csv"
countriesDf = pd.read_csv(path)
# a
x = scipy.stats.norm.rvs(size=30, loc=5, scale=1)
print(x)
# b
y = 5 * x + 2
print(y)
#c
def calcR(x,y):
    r=np.cov(x,y, ddof=0)[0][1]/(np.std(x, ddof=0)*np.std(y, ddof=0))

    return r
r= calcR(x,y)
print(r)
#d
def calcB(x,y,r):
    b=r*(np.std(y, ddof=0)/np.std(x, ddof=0))

    return b
print(calcB(x,y,r))

#e
noise= scipy.stats.norm.rvs(loc=0, scale=1, size=30)
newY=y+noise
r=calcR(x,newY)
print(r)
print(calcB(x,newY,r))

#f
noises=np.linspace(0.5,10,100)
rVector=[]
bVector = []
for noise in noises:
    noise = scipy.stats.norm.rvs(loc=0, scale=noise, size=30)
    newY= y + noise
    r = calcR(x, newY)
    b=calcB(x, newY, r)
    rVector.append(r)
    bVector.append(b)
print (rVector)
print (bVector)
plt.scatter(noises,rVector)
plt.show()
plt.scatter(noises,bVector)
plt.show()
