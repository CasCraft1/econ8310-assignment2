import pandas as pd
import numpy as np
import os
from datetime import time, datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier




#os.chdir(r'C:\Users\mackm\Documents\Other\School\UNO\Semester 3\Forecasting\assignment2')

df =  pd.read_csv('assignment2train.csv')


df['DateTime'] = pd.to_datetime(df['DateTime'])
df['hour'] =  df['DateTime'].dt.hour


y = df['meal']
x = df.drop(['meal', 'DateTime', 'id'], axis=1)




model = XGBClassifier(n_estimators=50, max_depth=3,
learning_rate=0.5, objective='binary:logistic')


modelFit = model.fit(x,y)

#same thing but for training set

testing = pd.read_csv('assignment2test.csv')

testing['DateTime'] = pd.to_datetime(df['DateTime'])
testing['hour'] =  testing['DateTime'].dt.hour


yt = testing['meal']
xt = testing.drop(['meal', 'DateTime', 'id'], axis=1)

pred = modelFit.predict(xt)


