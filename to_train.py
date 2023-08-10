# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from tensorflow.keras import models,layers,optimizers,regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score # R square
import joblib
import matplotlib.pyplot as plt

EPOCH=10

data=pd.read_csv("data3.csv",encoding='gbk')
data=data.drop(["房名","链接"],axis=1)
value=np.array(data.loc[:,"租金 (元/月)"])
data=np.array(data.drop("租金 (元/月)",axis=1))
"""缺失值处理，其实也可以删了，因为补充了缺失值
   不过np可以直接处理nan，这一点好棒"""
nan_row=np.where(np.isnan(data))[0]
nan_column=np.where(np.isnan(data))[1]

for i in range(len(nan_column)):
    data[nan_row[i]][nan_column[i]]=0
    
data_big=np.concatenate((np.reshape(data[:,0],(len(data[:,0]),1)),np.reshape(data[:,16],(len(data[:,16]),1)),data[:,42:]),axis=1)
data_one_hot=np.concatenate((data[:,1:16],data[:,17:42]),axis=1)
#归一化
scale=MinMaxScaler()
data_big=scale.fit_transform(data_big)

data=np.concatenate((data_big,data_one_hot),axis=1)

#l的是神经元数目，gamma是正则化参数，lr是学习率，w1是卷积窗口大小，w2是池化窗口大小
def cnn(l1=512,l2=1024,gamma=1e-4,lr=1e-3,w1=3,w2=2):
     #序列化输入
     model=models.Sequential()
     
     #1D用于处理文本或数值
     #卷积层用relu，池化层不用激活函数
     #l1输出空间维度，卷积中滤波器输出数目
     #l2是1D卷积窗口的长度
     model.add(layers.Conv1D(l1,w1,activation='relu',kernel_regularizer=
                             regularizers.l1(gamma),
                             input_shape=(5,10),padding='same'))
     
     #池化层
     model.add(layers.MaxPooling1D(w2))
     
     #拉长，因为特征图不能做乘法
     model.add(layers.Flatten())
     
     #七伤拳，避免过拟合
     model.add(layers.Dropout(0.5))
     
     #12层，通过L1正则化方式以消除系统过拟合。
     #L1正则化的效果是让尽可能多的权重变为零，非零权重的取值也有一定的限制，从而简化模型
     #relu是应用最广的激活函数
     model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
     model.add(layers.Dense(2*l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
     model.add(layers.Dense(3*l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
     model.add(layers.Dense(4*l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
     model.add(layers.Dense(5*l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
     model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
     model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
     model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
     model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
     model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
     model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
     model.add(layers.Dense(1,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
     
     #adam优化器，学习率=le-3(初始学习不快但能收敛到更好的性能)
     #一阶矩估计的指数衰减率0.9，二阶矩估计的指数衰减率0.999（在系数梯度中接近1更好），epsilon是防止在实现中除以0
     adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
     model.compile(optimizer=adam, loss="mse", metrics=['acc'])
     return model
 
train_data,test_data,train_value,test_value=train_test_split(data,value,random_state=0,train_size=0.9)
train_data=np.reshape(train_data,(len(train_data),5,10))
test_data=np.reshape(test_data,(len(test_data),5,10))

model=cnn()
history=model.fit(train_data,train_value,batch_size=1024,epochs=EPOCH,verbose=1,shuffle=True,validation_data=(test_data,test_value))
test_preidct=model.predict(test_data).reshape(len(test_data))

"""模型评估"""
mse=mean_squared_error(test_preidct,test_value)
rmse=np.sqrt(mse)
mae=mean_absolute_error(test_preidct,test_value)
r2=r2_score(test_preidct,test_value)

print("MSE:%lf\nRMSE:%lf\nMAE:%lf\nr2:%lf"%(mse,rmse,mae,r2))

"""损失值"""
train_loss=history.history["loss"]
val_loss=history.history["val_loss"]

plt.figure()
plt.title("loss_value in each epoch")
plt.style.use('ggplot')

plt.plot(np.arange(1,EPOCH+1,1),train_loss,label="train_loss",color="blue",
         linestyle = '-',linewidth=1)
plt.plot(np.arange(1,EPOCH+1,1),val_loss,label="val_loss",color="red",
         linestyle = '--',linewidth=1)

plt.xlabel("EPOCH")
plt.ylabel("loss_value")

plt.legend()

plt.show()

"""模型保存"""
model.save("train_modelwithoutjignwei.h5")



