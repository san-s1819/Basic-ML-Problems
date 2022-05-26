# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 19:20:00 2020

@author: Sanjay
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%%
train= pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

Y=train.iloc[:,-1].values
res=test.iloc[:,0]

#dropped columns with too many null vlaues
train.drop(['FireplaceQu','Id','PoolQC','Fence','Alley','MiscFeature'],axis=1,inplace=True)
test.drop(['FireplaceQu','Id','PoolQC','Fence','Alley','MiscFeature'],axis=1,inplace=True)

corrmat = train.corr()
#f, ax = plt.subplots(figsize=(12, 9))
#sns.heatmap(corrmat, vmax=.8, square=True);

k = 10 #number of variables for heatmap

#for showing only the correlations that matter not everything
cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index

#for calculating correlation coefficients as input to heatmap
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

train.drop('SalePrice',axis=1,inplace=True)

#cols=list(train.columns.values)
#combining train and test set to do preprocessing
train = pd.concat([train, test], axis=0)
#MISSING VALUES
#Filling missing categorical data with mode of data
columns = ['BsmtQual', 'GarageYrBlt', 'GarageType', 'GarageCond',
           'GarageFinish', 'GarageQual', 'MasVnrType', 'MasVnrArea',
           'BsmtExposure','BsmtFinType2',
           'BsmtFinType1', 'Utilities','BsmtFullBath', 'BsmtHalfBath',
           'Functional', 'SaleType', 'Exterior2nd',
           'Exterior1st', 'KitchenQual','BsmtCond',
           'Electrical','MSZoning']

#filling missing categorical data with the mode of the data
for item in columns:
    train[item] = train[item].fillna(train[item].mode()[0])


columns1 = ['GarageCars', 'BsmtFinSF1',
            'BsmtFinSF2', 'BsmtUnfSF',  'TotalBsmtSF', 'GarageArea','LotFrontage']

#filling missing numerical data with mean of the values
for item in columns1:
    train[item] = train[item].fillna(train[item].mean())

#correlation matrix
# corrmat = train.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True);

# sns.heatmap(train.isnull(),yticklabels=False, )
train['LotFrontage'] = train['LotFrontage'].fillna(train.LotFrontage.mean())

cols = ['MSZoning', 'Street',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
       'KitchenQual', 'Functional', 'GarageType', 'GarageFinish',
       'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']


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
X=df_encoded.iloc[:1460,:-1].values


#%%

#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

#X_train = np.append (arr=np.ones([X_train.shape[0],1]).astype(int), values = X_train, axis = 1)
#X_opt=[]
#for i in range(0,231):
#    X_opt.append(i)
#X_opt.remove(87)
#X_opt.remove(93)

#import statsmodels.formula.api as sm
#import statsmodels.regression.linear_model as sm
#from sklearn.tree import DecisionTreeRegressor
#dec_tree = DecisionTreeRegressor(random_state = 0)

#regressor = sm.OLS(Y_train, X_train[:,X_opt]).fit()
#print(regressor.summary())
#%%
# train=df_encoded.iloc[:1460,:]
# test=df_encoded.iloc[1460:,:]

# X_train =train.iloc[:,:].values
# X_test =test.iloc[:,:].values
X_test=df_encoded.iloc[1461:,:].values


from sklearn.tree import DecisionTreeRegressor
dec_tree = DecisionTreeRegressor(random_state = 0)
dec_tree.fit(X, Y)
y_pred4 = dec_tree.predict(X_test)
df = pd.DataFrame(dict(Id = res, SalePrice = y_pred4)).reset_index()
df.to_csv('Decsion_tree',index=False)

print(dec_tree.score(X_test,Y_test))


#%%
# #%% Model fitting and writing predictions to a file

# #Linear Regression

# from sklearn.linear_model import LinearRegression
# regressor_1 = LinearRegression()
# regressor_1.fit(X_train, Y_train)
# y_pred1 = regressor_1.predict(X_test)

# df = pd.DataFrame(dict(Id = res, SalePrice = y_pred1)).reset_index()
# df.drop('index',axis=1,inplace=True)
# df.to_csv('linear_housing.csv',index=False)

# #Polynomial Regression
# # from sklearn.preprocessing import PolynomialFeatures
# # poly_reg = PolynomialFeatures(degree = 4)
# # X_poly = poly_reg.fit_transform(X_train)
# # poly_reg.fit(X_poly, Y_train)
# # poly_reg = LinearRegression()
# # poly_reg.fit(X_poly, Y_train)
# # y_pred2 = poly_reg.predict(X_test)

# # df = pd.DataFrame(dict(Id = res, SalePrice = y_pred2)).reset_index()
# # df.drop('index',axis=1,inplace=True)
# # df.to_csv('poly_reg_housing.csv',index=False)


# #Support Vector Regression Linear Kernel

# from sklearn.svm import SVR
# svr = SVR(kernel = 'linear')
# svr.fit(X_train, Y_train)
# y_pred2 = svr.predict(X_test)

# df = pd.DataFrame(dict(Id = res, SalePrice = y_pred2)).reset_index()
# df.drop('index',axis=1,inplace=True)
# df.to_csv('linear_svr_housing.csv',index=False)

# #Support Vecor Regression Gaussian kernel

# from sklearn.svm import SVR
# svr_g = SVR(kernel = 'rbf')
# svr_g.fit(X_train, Y_train)
# y_pred3 = svr_g.predict(X_test)

# df = pd.DataFrame(dict(Id = res, SalePrice = y_pred3)).reset_index()
# df.drop('index',axis=1,inplace=True)
# df.to_csv('gaussian_svr_housing.csv',index=False)

# #Decision tree regsression
# from sklearn.tree import DecisionTreeRegressor
# dec_tree = DecisionTreeRegressor(random_state = 0)
# dec_tree.fit(X_train, Y_train)
# y_pred4 = dec_tree.predict(X_test)

# df = pd.DataFrame(dict(Id = res, SalePrice = y_pred4)).reset_index()
# df.drop('index',axis=1,inplace=True)
# df.to_csv('decision_tree_housing.csv',index=False)

# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators =1000, random_state = 0)
# regressor.fit(X_train, Y_train)
# y_pred5 = regressor.predict(X_test)

# df = pd.DataFrame(dict(Id = res, SalePrice = y_pred5)).reset_index()
# df.drop('index',axis=1,inplace=True)
# df.to_csv('random_forest_housing_6.csv',index=False)
