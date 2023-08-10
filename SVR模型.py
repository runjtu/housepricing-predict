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
url = r'D:/lasso2.csv'
df_sc = pd.read_csv(url)
y = df_sc['租金']
X = df_sc.drop('租金', axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scaler=StandardScaler()
scaler=scaler.fit(x_train)
x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)
y=y_train
X=x_train_scaled


# In[2]:


svr_rbf = SVR(kernel='rbf', C=100, gamma=0.01)
svr_lin = SVR(kernel='linear', C=100)
svr_poly = SVR(kernel='poly', C=100, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(x_test_scaled)
y_lin = svr_lin.fit(X, y).predict(x_test_scaled)
y_poly = svr_poly.fit(X, y).predict(x_test_scaled)


# In[3]:


msetest=mean_squared_error(y_rbf,y_test)
print(msetest)


# In[19]:


msetest=mean_squared_error(y_poly,y_test)
print(msetest)


# In[21]:


msetest=mean_squared_error(y_lin,y_test)
print(msetest)


# In[23]:


svr_poly = SVR(kernel='poly', C=100, degree=2)
y_lin = svr_lin.fit(X, y).predict(x_test_scaled)
print(y_poly)


# In[4]:


from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print("svr")
print("mean_absolute_error:", mean_absolute_error(y_rbf,y_test))
print("mean_squared_error:", mean_squared_error(y_rbf,y_test))
print("rmse:", sqrt(mean_squared_error(y_rbf,y_test)))
print("r2 score:", r2_score(y_rbf,y_test)) 


# In[ ]:




