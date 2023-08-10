# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
model = tf.keras.models.load_model("train_model.h5")
#separator
data_old=pd.read_csv("data.csv",encoding='gbk')
data_old=data_old.drop(["房名","链接"],axis=1)
value=np.array(data_old.loc[:,"租金 (元/月)"])
data_old=np.array(data_old.drop("租金 (元/月)",axis=1))
"""缺失值处理"""
nan_row=np.where(np.isnan(data_old))[0]
nan_column=np.where(np.isnan(data_old))[1]

for i in range(len(nan_column)):
    data_old[nan_row[i]][nan_column[i]]=0
    
data_big_old=np.concatenate((np.reshape(data_old[:,0],(len(data_old[:,0]),1)),np.reshape(data_old[:,16],(len(data_old[:,16]),1)),data_old[:,42:]),axis=1)
data_one_hot_old=np.concatenate((data_old[:,1:16],data_old[:,17:42]),axis=1)
#x=input()
x=[1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,35,0,	1,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	0,	0,	0,	108.9382,	34.33184,42.86,904	,180,1,4,0]
data=np.array(eval('x')).astype("float32").reshape((1,50))

data_big_new=np.concatenate((np.reshape(data[:,0],(len(data[:,0]),1)),np.reshape(data[:,16],(len(data[:,16]),1)),data[:,42:]),axis=1)
data_big_new=np.concatenate((data_big_old,data_big_new),axis=0)

data_one_hot_new=np.concatenate((data[:,1:16],data[:,17:42]),axis=1)
data_one_hot_new=np.concatenate((data_one_hot_old,data_one_hot_new),axis=0)

scale=MinMaxScaler()
data_big_new=scale.fit_transform(data_big_new)

data=np.concatenate((data_big_new,data_one_hot_new),axis=1)

data_use=np.reshape(data[-1],(1,5,10))
predict_value=model.predict(data_use)
print("预测的价格为:%lf(元/月)"%predict_value)