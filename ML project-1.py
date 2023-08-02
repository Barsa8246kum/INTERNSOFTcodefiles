import pandas as pd
import matplotlib.pyplot as plt



#Reading the data from your file
data = pd.read_csv('advertising.csv')
data.head()


#To visualize the data
fig ,axs = plt.subplots(1,3,sharey=True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(14,7))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])


#Creating X&Y for Linear Regression
feature_cols=['TV']
x=data[feature_cols]
y=data.Sales


#Importing Linear Regression Algo
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)

print(lr.intercept_)
print(lr.coef_)

result=6.97482+0.055*50
print(result)


#Create a DataFrame with min & max vaues of the Table
X_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()



preds = lr.predict(X_new)
preds

data.plot(kind='scatter',x='TV',y='Sales')

plt.plot(X_new,preds,c='red',linewidth=3)



import statsmodels.formula.api as smf
lm= smf.ols(formula='Sales~TV',data=data).fit()
lm.conf_int()



# finding the probablity values
lm.pvalues



# Finding the R-Squared values
lm.rsquared


# Multi LinearRegression
feature_cols=['TV','Radio','Newspaper']
x=data[feature_cols]
y=data.Sales


lr =  LinearRegression()
lr.fit(x,y)


print(lr.intercept_)
print(lr.coef_)



lm= smf.ols(formula='Sales~TV+Radio+Newspaper',data=data).fit()


lm.conf_int()
lm.summary()
