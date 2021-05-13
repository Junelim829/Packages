#주제분석 package

import pandas as pd
import numpy as np

# Chapter 1. 데이터 분할
#1.1 데이터 불러오기
train = pd.read_csv('C:/Users/Jooeun/Desktop/train.csv')

#1.2 단위 수정
train_x = train.drop('price', axis=1)
train_y = train['price']

#1.3 validation set 분할 - 첫번째
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train_x,train_y, test_size=0.2, train_size=0.8)

#1.4 질문
print('이렇게 데이터를 나누게 된다면 미래으 데이터로 과거의 데이터를 추정하게 된다.그래서 test error를 올바르게 추정할 수 없다.')

#1.5 validation set 분할 - 두번째
val = train[train.transaction_year == 4]
tr = train[train.transaction_year != 4]
val_x = val.drop(columns=['price'])
val_y = val['price']
tr_x = tr.drop(columns=['price'])
tr_y = tr['price']

#1.6
from category_encoders.cat_boost import CatBoostEncoder

cbe_encoder=CatBoostEncoder()
cbe_encoder.fit(tr_x,tr_y)
tr_cbe=cbe_encoder.fit_transform(tr_x,tr_y)
val_cbe=cbe_encoder.transform(val_x)

val_cbe.head()


#2. Ridge Regression (모범답안 참고)
#2.1 상관계수 플랏
import matplotlib.pyplot as plt
import seaborn as sns 

plt.figure(figsize=(10,8))
sns.heatmap(data = tr_cbe.corr(), annot=True, fmt = '.2f', linewidths=.5)
plt.show()

#2.2
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from math import sqrt

alphas=[ 0.00001, 0.0001, 0.001, 0.01, 0.1]
MSE = [0,0,0,0,0]
for i in range(5):
    alpha=alphas[i]
    ridge = Ridge(alpha, normalize=True)
    ridge.fit(tr_cbe,tr_y)
    pred = ridge.predict(val_cbe)
    MSE[i] = sqrt(mean_squared_error(pred, val_y))

#2.5
from sklearn.linear_model import LinearRegression
line_regression = LinearRegression()
line_regression.fit(tr_cbe, tr_y)
pred= line_regression.predict(val_cbe)
reg_mse=sqrt(mean_squared_error(pred, val_y))
reg_mse<min(mse)


#Chapter 3. LightGBM 
#3.1

#3.2
from sklearn.model_selection import KFold,GridSearchCV
import lightgbm as lgb
from lightgbm import LGBMModel,LGBMRegressor

rates=[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.99]
MSE = [0,0,0,0,0,0,0]
for i in range(7):
    rate=rates[i]
    lgbm =LGBMRegressor(learning_rate=rate)
    lgbm.fit(tr_cbe,tr_y)
    pred = lgbm.predict(val_cbe)
    MSE[i] = sqrt(mean_squared_error(pred, val_y))


#Chapter 4. test set
#4.1
test = pd.read_csv('C:/Users/Jooeun/Desktop/test.csv')
test_x = test.drop(columns=['price'])
test_y = test['price']

#4.2 캣부스트 인코딩
cbe_encoder=CatBoostEncoder()
cbe_encoder.fit(tr_x,tr_y)
tr_cbe=cbe_encoder.fit_transform(tr_x,tr_y)
te_cbe=cbe_encoder.transform(te_x)
te_cbe.head()


