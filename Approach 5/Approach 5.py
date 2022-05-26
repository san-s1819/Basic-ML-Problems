# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:53:06 2022

@author: Sanjay
"""

#%%Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
#Visuling null values
import missingno as msno
import xgboost as xgb
import category_encoders as ce

#%% Reading from files
train= pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
Y_train=train.iloc[:,-1].values
res=test.iloc[:,0]


#Handling null values and imputing NA
#%%Dropping columns with lot of null values/Useless columns
train.drop(['FireplaceQu','Id','Utilities','PoolQC','Fence','Alley','MiscFeature','SalePrice','MasVnrArea','MasVnrType'],axis=1,inplace=True)
test.drop(['FireplaceQu','Id','Utilities','PoolQC','Fence','Alley','MiscFeature','MasVnrArea','MasVnrType'],axis=1,inplace=True)

#{'col':'Utilities','mapping':{'AllPub':1,'NoSewr':2,'NoSeWa':3,'ELO':4}},
#{'col':'PoolQC','mapping':{'Ex':1,'Gd':2,'TA':3,'Fa':4,'NA':5}}
#{'col':'FireplaceQu','mapping':{'Ex':1,'Gd':2,'TA':3,'Fa':4,'Po':5,'NA':6}}

#Ordinal data encoding
mappingCols=[{'col':'LotShape','mapping':{'Reg':1,'IR1':2,'IR2':3,'IR3':4}}
         ,{'col':'LandContour','mapping':{'Lvl':1,'Bnk':2,'HLS':3,'Low':4}},
         {'col':'LandSlope','mapping':{'Gtl':1,'Gtl':2,'Sev':3}},
         {'col':'ExterQual','mapping':{'Ex':1,'Gd':2,'TA':3,'Fa':4,'Po':5}},
         {'col':'ExterCond','mapping':{'Ex':1,'Gd':2,'TA':3,'Fa':4,'Po':5}},
         {'col':'BsmtQual','mapping':{'Ex':1,'Gd':2,'TA':3,'Fa':4,'Po':5,'NA':6}},
         {'col':'BsmtCond','mapping':{'Ex':1,'Gd':2,'TA':3,'Fa':4,'Po':5,'NA':6}},
         {'col':'BsmtExposure','mapping':{'Gd':1,'Av':2,'Mn':3,'No':4,'NA':5}},
         {'col':'BsmtFinType1','mapping':{'GLQ':1,'ALQ':2,'Rec':3,'BLQ':4,'LwQ':5,'Unf':6,'NA':7}},
         {'col':'BsmtFinType2','mapping':{'GLQ':1,'ALQ':2,'Rec':3,'BLQ':4,'LwQ':5,'Unf':6,'NA':7}},
         {'col':'HeatingQC','mapping':{'Ex':1,'Gd':2,'TA':3,'Fa':4,'Po':5}},
         {'col':'KitchenQual','mapping':{'Ex':1,'Gd':2,'TA':3,'Fa':4,'Po':5}},
         {'col':'GarageFinish','mapping':{'Fin':1,'RFn':2,'Unf':3,'NA':4}},
         {'col':'GarageCond','mapping':{'Ex':1,'Gd':2,'TA':3,'Fa':4,'Po':5,'NA':6}},
         {'col':'GarageQual','mapping':{'Ex':1,'Gd':2,'TA':3,'Fa':4,'Po':5,'NA':6}},
         {'col':'PavedDrive','mapping':{'Y':1,'P':2,'N':3}}]

encoder= ce.OrdinalEncoder(cols=['LotShape','LandContour','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond',
                                 'BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','KitchenQual','FireplaceQu',
                                 'GarageFinish','GarageCond','GarageQual','PavedDrive'],return_df=True,
                           mapping=mappingCols)
train=encoder.fit_transform(train)
test=encoder.fit_transform(test)

#Helpful code for checking null values
#list of columns which have null value
#test.columns[test.isna().any()].tolist()
#df.isnull().sum()
#df.isnull().sum(axis=1) row wise null values
#train.isnull().sum(axis=1).sort_values(ascending=False) sort nullvalues

#visualising null values
#msno.bar(train)
#msno.bar(test)
#msno.matrix(train)
#msno.matrix(train)
#msno.heatmap(train)

#Preprocessing and handling null values
train['Age']=2011-train['YearBuilt']
train.drop(['YearBuilt'],axis=1,inplace=True)
train['YrSold']=2011-train['YrSold']
train['GarageAge']=2011-train['GarageYrBlt']
train.drop(['GarageYrBlt'],axis=1,inplace=True)
train['GarageAge'].fillna(0.0,inplace=True)
train['YearRemodAdd']=2011-train['YearRemodAdd']

test['Age']=2011-test['YearBuilt']
test.drop(['YearBuilt'],axis=1,inplace=True)
test['YrSold']=2011-test['YrSold']
test['GarageAge']=2011-test['GarageYrBlt']
test.drop(['GarageYrBlt'],axis=1,inplace=True)
test['GarageAge'].fillna(0.0,inplace=True)
test['YearRemodAdd']=2011-test['YearRemodAdd']

train['LotFrontage'].fillna(train['LotFrontage'].median(),inplace=True)
test['LotFrontage'].fillna(train['LotFrontage'].median(),inplace=True)




#filling categorical null values

# train['GarageCond'].fillna(train['GarageCond'].mode()[0],inplace=True)
# test['GarageCond'].fillna(test['GarageCond'].mode()[0],inplace=True)

train['GarageArea'].fillna(train['GarageArea'].median(),inplace=True)
test['GarageArea'].fillna(test['GarageArea'].median(),inplace=True)

# train['GarageQual'].fillna(train['GarageQual'].mode()[0],inplace=True)
# test['GarageQual'].fillna(test['GarageQual'].mode()[0],inplace=True)

# train['BsmtQual'].fillna(train['BsmtQual'].mode()[0],inplace=True)
# test['BsmtQual'].fillna(test['BsmtQual'].mode()[0],inplace=True)

# train['BsmtCond'].fillna(train['BsmtCond'].mode()[0],inplace=True)
# test['BsmtCond'].fillna(test['BsmtCond'].mode()[0],inplace=True)

# train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0],inplace=True)
# test['BsmtExposure'].fillna(test['BsmtExposure'].mode()[0],inplace=True)

# train['BsmtFinType1'].fillna(train['BsmtFinType1'].mode()[0],inplace=True)
# test['BsmtFinType1'].fillna(test['BsmtFinType1'].mode()[0],inplace=True)

# train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0],inplace=True)
# test['BsmtFinType2'].fillna(test['BsmtFinType2'].mode()[0],inplace=True)

train['Electrical'].fillna(train['Electrical'].mode()[0],inplace=True)

train['GarageType'].fillna(train['GarageType'].mode()[0],inplace=True)
test['GarageType'].fillna(test['GarageType'].mode()[0],inplace=True)

# train['GarageFinish'].fillna(train['GarageFinish'].mode()[0],inplace=True)
# test['GarageFinish'].fillna(test['GarageFinish'].mode()[0],inplace=True)

train['BsmtFinSF1'].fillna(train['BsmtFinSF1'].median(),inplace=True)
train['BsmtFinSF2'].fillna(train['BsmtFinSF2'].median(),inplace=True)


test['MSZoning'].fillna(test['MSZoning'].mode()[0],inplace=True)
test['Exterior1st'].fillna(test['Exterior1st'].mode()[0],inplace=True)
test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0],inplace=True)
test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].median(),inplace=True)
test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].median(),inplace=True)
test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].median(),inplace=True)
test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].median(),inplace=True)
test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0],inplace=True)
test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0],inplace=True)
#test['KitchenQual'].fillna(test['KitchenQual'].mode()[0],inplace=True)
test['Functional'].fillna(test['Functional'].mode()[0],inplace=True)
test['GarageCars'].fillna(test['GarageCars'].mode()[0],inplace=True)
test['SaleType'].fillna(test['SaleType'].mode()[0],inplace=True)

train = pd.concat([train, test], axis=0)

cols = ['MSZoning', 'Street','LotConfig',
'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
'RoofMatl', 'Exterior1st', 'Exterior2nd',
'Foundation','Heating','CentralAir', 'Electrical',
'Functional', 'GarageType', 'SaleType', 'SaleCondition']


def One_hot_encoding(columns):
    df_final=train
    i=0
    for fields in columns:

        df1=pd.get_dummies(train[fields],drop_first=True)
        #print(df1)
        train.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1


    df_final=pd.concat([train,df_final],axis=1)

    return df_final

df_encoded=One_hot_encoding(cols)

train=df_encoded.iloc[:1460,:]
test=df_encoded.iloc[1460:,:]
#%%EDA
sns.distplot(df_train['SalePrice']);
sns.distplot(train['LotFrontage']);
sns.distplot(train['LotArea']);
print("Skewness: %f" % train['SalePrice'].skew())
print("Skewness: %f" % train['SalePrice'].kurt())

var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

plt.scatter(x=train["GrLivArea"],y=train["SalePrice"])
plt.xlabel("Living Area")
plt.ylabel("Sale price")
plt.show()

#box plot overallqual/saleprice
var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);

#%% Feature Selection
#Heatmap with no correlation coefficient values
corrmat = train.corr()
hm = sns.heatmap(corrmat, cbar=True, vmax=0.8, square=True)

#Heatmap with 10 largest correlation coefficient values
corrmat = train.corr()
cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
#for calculating correlation coefficients as input to heatmap
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=0.75)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], height = 2.5)
plt.show()

#%% Fitting random forest model to data
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators =1000, random_state = 0)
regressor.fit(train, Y_train)
y_pred5 = regressor.predict(test)

df = pd.DataFrame(dict(Id = res, SalePrice = y_pred5)).reset_index()
df.drop('index',axis=1,inplace=True)
df.to_csv('random_forest_housing_9.csv',index=False)

# xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 10)

# xg_reg.fit(train,Y_train)
# preds = xg_reg.predict(test)

# duplicate_columns = test_df.columns[test_df.columns.duplicated()]
