'''주제분석 3주차 패키지 - 임주은'''

import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from category_encoders.cat_boost import CatBoostEncoder
import lightgbm as lgb
from lightgbm import LGBMModel,LGBMRegressor

train = pd.read_csv('C:/Users/Jooeun/Desktop/주제분석 3주차 패키지/train.csv')
test=  pd.read_csv('C:/Users/Jooeun/Desktop/주제분석 3주차 패키지/test.csv')

train_x = train.drop(['price'], axis = 1)
train_y = train.loc[:, ['price']]
test_x = test.drop(['price'], axis = 1)
test_y = np.sqrt(test.loc[:, ['price']])

feature_list = list(train_x.columns)

CBE_encoder = CatBoostEncoder()
train_cbe = CBE_encoder.fit_transform(train_x[feature_list], train_y)
test_cbe = CBE_encoder.transform(test_x[feature_list])

best_lgbm_reg = LGBMRegressor(learning_rate = 0.3)
best_lgbm_reg.fit(train_cbe, train_y)


#1.1 모델 불러오기
import pickle
import joblib
lgbm_data = joblib.load('lgbm.pkl')


#1.2 Feature Importance 확인과 해석
from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(12,10))
plot_importance(lgbm_data, ax=ax)
plt.show()
print('해석: pickle이 설치되지 않아 해석하기 어렵지만 plot_importance에 대해 대신 설명하도록 하겠습니다.\
  plot importance 는 feature importance를 보여주는데, feature importance는 랜덤 포레스트 모델을 이용할 때 참고했던 지표이다.\
  feature importance는 다소 biased하다는 문제점이 있다. 그 이유는 연속형 변수, high cardinality 변수들의 중요도가\
  부풀어 나타날 수 있기 때문이다. 노드 중요값이 높게 나오는 것은 cardinality가 클수록 더 잘 분해해야하기 때문일 것이다.')


#1.3 Randomness Control
import random
import os
def seed_everything(seed: int=42):
  random.seed(seed)
  os.seed(seed)
  np.random.seed(seed)



#1.4 Permutation Feature Importance
from sklearn.inspection import permutation_importance



#1.5 SHAP(SHapley Additive exPlanations)의 확인과 해석
import shap

print('shap가 설치되지 않아 확인과 해석이 어렵지만')






#2.1 주어진 코드를 시행하고, 현재 신경망의 구조에 대해 간단히 설명해주세요.
train = pd.read_csv('C:/Users/Jooeun/Desktop/주제분석 3주차 패키지/train.csv')
train_x = train.drop(['price'], axis = 1)
train_y = train.loc[:, ['price']]

val_x = train_x[train['transaction_year'] == 4]
val_y = train_y[train['transaction_year'] == 4]
train_tune_x = train_x[train['transaction_year'] < 4]
train_tune_y = train_y[train['transaction_year'] < 4]

CBE_encoder = CatBoostEncoder()
train_tune_cbe = CBE_encoder.fit_transform(train_tune_x[feature_list], train_tune_y)
val_cbe = CBE_encoder.transform(val_x[feature_list])

import tensorflow as tf
import keras

from keras import models
from keras import layers
from keras.layers import Dense
from keras.optimizers import Adam

def build_model():
  model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=[len(train_tune_cbe.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.0001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mse'])
  return model

model = build_model()
model.summary()

#2.2 training loss와 validation loss를 시각화하세요.
history = model.fit(train_tune_cbe, train_tune_y, epochs=300, validation_data = (val_cbe, val_y), batch_size = 512)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

#2.3 어떤 문제가 발생했는지 설명해주세요.
print('같은 조원의 그래프를 참고해서 보았을 때 mse의 값이 \
단순히 값의 합산을 통해 평균을 내어, 실제 오차보다 훨씬 줄어든 양상을 보인다. ')


#2.4 더 좋은 성능의 딥러닝 모델을 위해
feature_list = list(train_x.columns)

CBE_encoder = CatBoostEncoder()
train_cbe = CBE_encoder.fit_transform(train_x[feature_list], train_y)
test_cbe = CBE_encoder.transform(test_x[feature_list])

print('패키지 설치에 계속 오류가 떠서 글로 대체한 후, 컴퓨터를 점검한 후에 다시 패키지 과제를 풀어보겠습니다!')
print('dense 조절은 keras.layers에 있는 dense함수를 이용하는 것으로 dense를 점차 줄여나가면서 성능을 좋게 만드는 방법이다.\
  이 방법은')
print('batch_size를 조절하면 전체 데이터를 쪼개 여러번 학습하면서 최소 요구되는 메모리의 양을 줄일 수 있다.\
  반면에 이를 늘리면 학습이 더 안정되는 모습을 확인할 수 있다. ')
print('regularization은 과적합을 막기위해 사용되는 방법으로 L1은 Feature selection이 가능하다는 특징이 있다.\
  L1 Regularization은 cost function에 가중치의 절댓값을 더해주는 것이며 이를 Lasso regression이라고 부른다. \
  L2 Regularization은 가중치의 제곱을 포함하여 더하는 방법이다. 이를 ridge regression 이라 부른다.')
print('dropout은 가중치의 크기를 패널티로 하여 가중치의 값이 강제로 작아지게 하는 방법이다.\
  연결을 끊어 남은 노드들만을 통해 훈련을 하는 것으로, 이 또한 오버피팅을 해결하기 위함이다.')
print('batch normalization은 학습의 효율을 높이는 방법으로 학습속도를 개선하고, 과적합의 위험을 줄인다는 장점이 있다.\
  또한 학습을 진행할 때마다 출력값을 정규화하기 때문에 초기 가중치 값 선택의 의존성이 적어진다.\
  dropout과 유사한 방법이다.')
