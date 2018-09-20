# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('sample_submission.csv')
store = pd.read_csv('store.csv')
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

def transform(data):
    #remove NaN
    data.fillna(0, inplace=True)
    
    #change the StateHoliday format 
    data.loc[data['StateHoliday'] == 'a', 'StateHoliday'] = '1'
    data.loc[data['StateHoliday'] == 'b', 'StateHoliday'] = '2'
    data.loc[data['StateHoliday'] == 'c', 'StateHoliday'] = '3'
    data['StateHoliday'] = data['StateHoliday'].astype(float)
    
    #change the StoreType format 
    data.loc[data['StoreType'] == 'a', 'StoreType'] = '1'
    data.loc[data['StoreType'] == 'b', 'StoreType'] = '2'
    data.loc[data['StoreType'] == 'c', 'StoreType'] = '3'
    data.loc[data['StoreType'] == 'd', 'StoreType'] = '4'
    data['StoreType'] = data['StoreType'].astype(float)
    
    #change the Assortment format 
    data.loc[data['Assortment'] == 'a', 'Assortment'] = '1'
    data.loc[data['Assortment'] == 'b', 'Assortment'] = '2'
    data.loc[data['Assortment'] == 'c', 'Assortment'] = '3'
    data['Assortment'] = data['Assortment'].astype(float)
    
    #creat new columns
    data['year'] = data.Date.apply(lambda x: x.split('-')[0])
    data['year'] = data['year'].astype(float)
    data['month'] = data.Date.apply(lambda x: x.split('-')[1])
    data['month'] = data['month'].astype(float)
    data['day'] = data.Date.apply(lambda x: x.split('-')[2])
    data['day'] = data['day'].astype(float)
    
    #creat new columns
    data.loc[data['PromoInterval'] == 0, 'PromoInterval'] = '0,0,0,0'
    data['month1'] = data.PromoInterval.apply(lambda x: x.split(',')[0])
    data.loc[data['month1'] == 'Jan', 'month1'] = 1
    data.loc[data['month1'] == 'Feb', 'month1'] = 2
    data.loc[data['month1'] == 'Mar', 'month1'] = 3
    data.loc[data['month1'] == 'Apr', 'month1'] = 4
    data.loc[data['month1'] == 'May', 'month1'] = 5
    data.loc[data['month1'] == 'Jun', 'month1'] = 6
    data.loc[data['month1'] == 'Jul', 'month1'] = 7
    data.loc[data['month1'] == 'Aug', 'month1'] = 8
    data.loc[data['month1'] == 'Sept', 'month1'] = 9
    data.loc[data['month1'] == 'Oct', 'month1'] = 10
    data.loc[data['month1'] == 'Nov', 'month1'] = 11
    data.loc[data['month1'] == 'Dec', 'month1'] = 12
    data['month1'] = data['month1'].astype(float)
    
    data['month2'] = data.PromoInterval.apply(lambda x: x.split(',')[1])
    data.loc[data['month2'] == 'Jan', 'month2'] = 1
    data.loc[data['month2'] == 'Feb', 'month2'] = 2
    data.loc[data['month2'] == 'Mar', 'month2'] = 3
    data.loc[data['month2'] == 'Apr', 'month2'] = 4
    data.loc[data['month2'] == 'May', 'month2'] = 5
    data.loc[data['month2'] == 'Jun', 'month2'] = 6
    data.loc[data['month2'] == 'Jul', 'month2'] = 7
    data.loc[data['month2'] == 'Aug', 'month2'] = 8
    data.loc[data['month2'] == 'Sept', 'month2'] = 9
    data.loc[data['month2'] == 'Oct', 'month2'] = 10
    data.loc[data['month2'] == 'Nov', 'month2'] = 11
    data.loc[data['month2'] == 'Dec', 'month2'] = 12
    data['month2'] = data['month2'].astype(float)
    
    data['month3'] = data.PromoInterval.apply(lambda x: x.split(',')[2])
    data.loc[data['month3'] == 'Jan', 'month3'] = 1
    data.loc[data['month3'] == 'Feb', 'month3'] = 2
    data.loc[data['month3'] == 'Mar', 'month3'] = 3
    data.loc[data['month3'] == 'Apr', 'month3'] = 4
    data.loc[data['month3'] == 'May', 'month3'] = 5
    data.loc[data['month3'] == 'Jun', 'month3'] = 6
    data.loc[data['month3'] == 'Jul', 'month3'] = 7
    data.loc[data['month3'] == 'Aug', 'month3'] = 8
    data.loc[data['month3'] == 'Sept', 'month3'] = 9
    data.loc[data['month3'] == 'Oct', 'month3'] = 10
    data.loc[data['month3'] == 'Nov', 'month3'] = 11
    data.loc[data['month3'] == 'Dec', 'month3'] = 12
    data['month3'] = data['month3'].astype(float)
    
    data['month4'] = data.PromoInterval.apply(lambda x: x.split(',')[3])
    data.loc[data['month4'] == 'Jan', 'month4'] = 1
    data.loc[data['month4'] == 'Feb', 'month4'] = 2
    data.loc[data['month4'] == 'Mar', 'month4'] = 3
    data.loc[data['month4'] == 'Apr', 'month4'] = 4
    data.loc[data['month4'] == 'May', 'month4'] = 5
    data.loc[data['month4'] == 'Jun', 'month4'] = 6
    data.loc[data['month4'] == 'Jul', 'month4'] = 7
    data.loc[data['month4'] == 'Aug', 'month4'] = 8
    data.loc[data['month4'] == 'Sept', 'month4'] = 9
    data.loc[data['month4'] == 'Oct', 'month4'] = 10
    data.loc[data['month4'] == 'Nov', 'month4'] = 11
    data.loc[data['month4'] == 'Dec', 'month4'] = 12
    data['month4'] = data['month4'].astype(float)
    
    return data

# merge train and test with store

train=pd.merge(train, store, on=['Store'])
test=pd.merge(test, store, on=['Store'])

# transform
train = train[train["Open"] != 0]
train=transform(train)
test=transform(test)

customers_means = train.groupby('Store').mean().Customers
customers_means.name = 'CustomersMean'

train = train.join(customers_means, on='Store')
test = test.join(customers_means, on='Store')

Y_train=train.iloc[:, 3].values

Xtrain=train.drop(['Date','Sales','PromoInterval','Customers'],axis=1)
X_train=Xtrain.iloc[:,:22].values

Xtest=test.drop(['Date','Id','PromoInterval'],axis=1)
X_test=Xtest.iloc[:,:22].values

#Random Forest
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, Y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

submission = pd.DataFrame({"Id": test["Id"], "Sales": y_pred})
submission.to_csv("submission.csv", index=False)
