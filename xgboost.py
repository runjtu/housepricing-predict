#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xgboost
import numpy
import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
# 载入数据集
url = r'D:/lasso2.csv'
df_sc = pd.read_csv(url)
y = df_sc['租金']
X = df_sc.drop('租金', axis=1)


# In[2]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scaler=StandardScaler()
scaler=scaler.fit(x_train)
x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)
reg=XGBRegressor(max_depth=7,n_estimators=2000,learning_rate=0.3,eta=0.05).fit(x_train_scaled,y_train)
print(reg.score(x_train_scaled,y_train))
pres=reg.predict(x_test_scaled)
msetest=mean_squared_error(pres,y_test)
print(msetest)


# In[3]:


from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print("xgboost")
print("mean_absolute_error:", mean_absolute_error(pres,y_test))
print("mean_squared_error:", mean_squared_error(pres,y_test))
print("rmse:", sqrt(mean_squared_error(pres,y_test)))
print("r2 score:", r2_score(pres,y_test)) 


# In[ ]:




