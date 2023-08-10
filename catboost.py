#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import lightgbm as lgb
url = r'D:/lasso2.csv'
df_sc = pd.read_csv(url)
y = df_sc['租金']
X = df_sc.drop('租金', axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
scaler=StandardScaler()
scaler=scaler.fit(x_train)
x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)


# In[8]:


model = CatBoostRegressor(max_depth=7,n_estimators=2000,learning_rate=0.1)
# Fit model
model.fit(x_train_scaled,y_train)
# Get predictions
preds = model.predict(x_test_scaled)
msetest=mean_squared_error(preds,y_test)
print(msetest)


# In[9]:


print(preds)


# In[10]:


print(y_test)


# In[11]:


rr=r2_score(preds,y_test)
print(rr)


# In[12]:


from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print("catboost")
print("mean_absolute_error:", mean_absolute_error(preds,y_test))
print("mean_squared_error:", mean_squared_error(preds,y_test))
print("rmse:", sqrt(mean_squared_error(preds,y_test)))
print("r2 score:", r2_score(preds,y_test))


# In[ ]:




