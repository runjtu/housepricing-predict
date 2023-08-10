#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import requests
import json
import csv
import pprint
import pandas as pd
import codecs
with open(r'C:\Users\user\Desktop\对应的经纬度2.csv', 'ab+')as fp:
    fp.write(codecs.BOM_UTF8)
f = open(r'C:\Users\user\Desktop\对应的经纬度2.csv','w',newline='', encoding='utf-8')
writer = csv.writer(f)
writer.writerow(['jingdu','weidu','ditiezhan','gongjiaozhan','yiyuan','jiaoyu','gouwuzhongxin'])
r'C:\Users\user\Desktop\对应的经纬度2.csv'
df= pd.read_csv("a.csv",encoding='utf-8')
df1 = pd.read_csv("b.csv",encoding='utf-8')
for index1,row1 in df1.iterrows():
    for index,row in df.iterrows():
        if ((df.iloc[index, 0]==df1.iloc[index1, 0])&(df.iloc[index, 1]==df1.iloc[index1, 1])):
            print(df1.iloc[index1, 0])
            long=df1.iloc[index, 0]
            lat=df1.iloc[index, 1]
            subway=df.iloc[index, 2]
            bus=df.iloc[index, 3]
            hospital=df.iloc[index, 4]
            educate=df.iloc[index, 5]
            shopping=df.iloc[index, 6]
            writer.writerow([long,lat,subway,bus,hospital,educate,shopping])
            break
f.close()

