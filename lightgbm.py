#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
url = r'D:/lasso2.csv'
df_sc = pd.read_csv(url)
y = df_sc['租金']
X = df_sc.drop('租金', axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
scaler=StandardScaler()
scaler=scaler.fit(x_train)
x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)


# In[4]:


my_model = lgb.LGBMRegressor(objective='regression', num_leaves=7, learning_rate=0.1, n_estimators=1500,
                             verbosity=2, max_depth=7)
my_model.fit(x_train_scaled, y_train, verbose=False)
predictions = my_model.predict(x_test_scaled)
msetest=mean_squared_error(predictions,y_test)
print(msetest)


# In[31]:


print(predictions)


# In[32]:


print(y_test)


# In[33]:


from sklearn.metrics import r2_score
rr=r2_score(predictions,y_test)
print(rr)


# In[5]:


from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print("lightgbm")
print("mean_absolute_error:", mean_absolute_error(predictions,y_test))
print("mean_squared_error:", mean_squared_error(predictions,y_test))
print("rmse:", sqrt(mean_squared_error(predictions,y_test)))
print("r2 score:", r2_score(predictions,y_test)) 


# In[ ]:




