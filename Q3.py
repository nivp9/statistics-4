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
age = ageAndTimeDf["Age"]
time = ageAndTimeDf["Time"]

plt.scatter(age,time)

Bls=(np.cov(age, time, ddof=0)[0][1])/np.var(age)
x = np.linspace(0,35,100)
y = x*Bls
plt.plot(x, y, '-r')

plt.show()

fixedAge =[]
fixedTime =[]
for i in range((len(age))):
    if time[i]<40:
        fixedAge.append(age[i])
        fixedTime.append(math.log(time[i],math.e))

plt.scatter(fixedAge, fixedTime)

Bls = ((np.cov(fixedAge, fixedTime, ddof=0)[0][1]) / np.var(fixedAge))
x = np.linspace(0, 35, 100)
y = x * Bls
plt.plot(x, y, '-r')
plt.show()





# def calcR(x,y):
#     r=np.cov(x,y, ddof=0)[0][1]/(np.std(x, ddof=0)*np.std(y, ddof=0))
#
#     return r
# r= calcR(x,y)
# print(r)
# #d
# def calcB(x,y,r):
#     b=r*(np.std(y, ddof=0)/np.std(x, ddof=0))
#
#     return b

##### countries
countriesDf = pd.read_csv(path)
lifeEx = countriesDf["life_expectancy"]
income = countriesDf["income"]
edu = countriesDf["education"]
income=income.apply(lambda x: math.log(x,math.e))   ## for trans
r = np.cov(income, lifeEx, ddof=0)[0][1] / (np.std(income, ddof=0) * np.std(lifeEx, ddof=0))
Bls=(np.cov(income, lifeEx, ddof=0)[0][1])/np.var(income)
Als = np.mean(lifeEx)-Bls*np.mean(income)
x = np.linspace(5,14,1000)
y = x*Bls+Als
plt.scatter(income,lifeEx)
plt.plot(x,y)
plt.show()

print(r**2)


r = np.cov(edu, lifeEx, ddof=0)[0][1] / (np.std(edu, ddof=0) * np.std(lifeEx, ddof=0))
Bls=(np.cov(edu, lifeEx, ddof=0)[0][1])/np.var(edu)
Als = np.mean(lifeEx)-Bls*np.mean(edu)
x = np.linspace(0,20,100)
y = x*Bls+Als
plt.scatter(edu,lifeEx)
plt.plot(x,y)
plt.show()

print(r**2)

