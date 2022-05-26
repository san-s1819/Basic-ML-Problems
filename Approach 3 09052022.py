# -*- coding: utf-8 -*-
"""
Created on Mon May  9 19:56:53 2022

@author: Sanjay
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing

#reading data from csv file
train= pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
Y_train=train.iloc[:,-1].values
res=test.iloc[:,0]

#drop columns with lots of NA values
train.drop(['FireplaceQu','PoolQC','Fence','Alley','MiscFeature','Utilities'],axis=1,inplace=True)
test.drop(['FireplaceQu','PoolQC','Fence','Alley','MiscFeature','Utilities'],axis=1,inplace=True)

#columns with high correlation
col_imp=['OverallQual','GrLivArea','GarageCars','GarageArea',
         'TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd',
         'YearBuilt','SalePrice']

col_test=['Id','OverallQual','GrLivArea','GarageCars','GarageArea',
         'TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd',
         'YearBuilt']

#df_temp2=test[col_imp]
#normalizing data
#df_normalized_test=preprocessing.normalize(test[col_imp])
#test[test['GarageArea']==0.00]
#test.drop(test['TotalBsmtSF']==0,axis=0)
df_temp=test[col_test]
df_temp['TotalBsmtSF']=df_temp['TotalBsmtSF'].fillna(0)
df_temp['GarageCars']=df_temp['GarageCars'].fillna(0)
df_temp['GarageArea']=df_temp['GarageArea'].fillna(0)
for i in df_temp.index:
    if(df_temp['TotalBsmtSF'][i]==0):
        df_temp['TotalBsmtSF'][i]=df_temp['TotalBsmtSF'].mean()
    if(df_temp['GarageCars'][i]==0):
        df_temp['GarageCars'][i]=1
    if(df_temp['GarageArea'][i]==0):
        df_temp['GarageArea'][i]=df_temp['GarageArea'].mean()


        #res.drop(index=i,inplace=True)
res=df_temp.iloc[:,0].values
df_temp=df_temp.iloc[:,1:]
#df_temp=df_temp[:,:-1]
#Y_train=df_temp[:,-1]
#df_temp.dropna(axis=0,how='any',inplace=True)
df_normalized_test=preprocessing.normalize(df_temp)




df_normalized_train=preprocessing.normalize(train[col_imp])

#splitting into train and test data
X_train=df_normalized_train[:,:-1]
#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(df_normalized_train,Y_train, test_size = 0.2, random_state = 0)


#predicting
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators =1000, random_state = 0)
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(df_normalized_test)

df = pd.DataFrame(dict(Id = res, SalePrice = y_pred)).reset_index()
df.drop('index',axis=1,inplace=True)
df.to_csv('random_forest_housing_7.csv',index=False)
