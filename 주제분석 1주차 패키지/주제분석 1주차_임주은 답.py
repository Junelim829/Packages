import numpy as np
import pandas as pd

#1. 데이터 전처리
#1.1 데이터 불러오기
df = pd.read_csv('C:/Users/Jooeun/Desktop/주제분석 1주차/data.csv')



#1.2 데이터 확인
len(df.index)
len(df.columns)
df.shape
df.info()
df.isnull()


#1.3 불필요한 행 삭제
df = df[df['city'] != '부산광역시']
df.reset_index(drop=True, inplace=True)


#1.4 변수 이름 바꾸기
df.rename(columns ={'transaction_real_price':'price'},inplace = True)
print(df)


#1.5 불필요한 변수 삭제
df = df.drop(["transaction_id","apartment_id","jibun","city"], axis = 1)
print(df)


#1.6 연/월 뽑기
import datetime
df['transaction_year'] = df['transaction_year_month'] // 100
df['transaction_month'] = df['transaction_year_month'] % 100
print(df)


#1.7 필요연도 뽑기
df = df[df['transaction_year'] >= 2012]
df.reset_index()
df.head()




#2. 데이터 시각화
import matplotlib.pyplot as plt
import seaborn as sns

#2.1 거래가격 분포 확인
firstplot = plt.figure(figsize=(20,10))

axes1 = firstplot.add_subplot(1, 2, 1)
axes2 = firstplot.add_subplot(1, 2, 2)

axes1.boxplot(df['price'], vert=False)
axes1.set_title('Boxplot of Price')
axes1.set_xlabel('Price')
axes1.yaxis.set_ticklabels([])

axes2.hist(df['price'], bins=50)
axes2.set_title('Histogram of Price')
axes2.set_xlabel('Price')
plt.show()

sns.set(font_scale=2)
fig, axs = plt.subplots(ncols=2, figsize=(18,10))

boxplot_data = df['price'].reset_index()
boxplot_data = boxplot_data.drop(columns=['index'])

sns.boxplot(ax=axs[0], data=boxplot_data, orient='h')
axs[0].set_title('Boxplot of Price')
axs[0].set_xlabel('Price')
axs[0].yaxis.set_ticklabels([])
axs[0].set_ylabel('')

sns.distplot(boxplot_data, bins=50)
axs[1].set_title('Histogram of Price')
axs[1].set_xlabel('Price')
axs[1].set_ylabel('')
plt.show()

print('결과 해석: 중앙값이 20만원 이하에 분포해 있으며 상당수의 매물들이 20만원 이하에서 거래되고 있음을 확인할 수 있다. ')


#2.2 거래연도 분포 확인 / 거래연도별 가격 분포 확인 (모범답안 참고)
c_by_year = pd.DataFrame(df['price'].groupby(df['transaction_year']).count())
c_by_year['transaction_year'] = c_by_year.index.values
c_by_year

p_by_year = pd.DataFrame(df['price'].groupby(df['transaction_year']).mean())
p_by_year['transaction_year'] = p_by_year.index.values
p_by_year

std_error = np.std(p_by_year['price'], ddof=1) / np.sqrt(len(p_by_year))

color = ['#5975A4', '#CC8963', '#5F9E6E', '#B55D60', '#857AAB', '#8D7866']

plot = plt.figure(figsize=(20,10))
axes1 = plot.add_subplot(1, 2, 1)
axes2 = plot.add_subplot(1, 2, 2)

axes1.bar(c_by_year['transaction_year'], c_by_year['price'], color = color)
axes1.set_xlabel('transaction_year', fontsize = 20)
axes1.set_ylabel('Count', fontsize = 20)
axes1.grid(axis = 'x')

axes2.bar(p_by_year['transaction_year'], p_by_year['price'], color = color)
axes2.errorbar(p_by_year['transaction_year'], p_by_year['price'], yerr = std_error, ls = 'None', color = 'k', lolims = True)
axes2.set_xlabel('transaction_year', fontsize = 20)
axes2.set_ylabel('Price', fontsize = 20)
axes2.grid(axis = 'x')

print('플랏 해석:시간의 경과에 따라 가격 또한 증가했음을 알 수 있음.')


#2.3 층 분포 확인 / 층별 가격 분포 확인
sns.set(font_scale=1.5)
fig, axs = plt.subplots(ncols=2, figsize=(18,10))

sns.countplot(ax=axs[0], data=df, x='floor')
axs[0].set_xlabel('floor')
axs[0].xaxis.set_ticklabels([])
axs[0].set_ylabel('Count')

sns.barplot(ax=axs[1], data=df, x='floor', y='price', ci=None)
axs[1].set_xlabel('floor')
axs[1].xaxis.set_ticklabels([])
axs[1].set_ylabel('Price')

plt.show()



#2.4 완공연도 분포 확인 / 완공연도별 가격 분포 확인
c_by_completion = pd.DataFrame(df['price'].groupby(df['year_of_completion'])
p_by_completion = pd.DataFrame(df['price'].groupby(df['year_of_completion'])

plot = plt.figure(figsize=(20,10))
axes1 = plot.add_subplot(1, 2, 1)
axes2 = plot.add_subplot(1, 2, 2)

axes1.bar(c_by_completion.index.values, c_by_completion['price'], color = ['salmon'], alpha = 0.8)
axes1.set_xlabel('year_of_completion', fontsize = 18)
axes1.set_ylabel('Count', fontsize = 18)
axes1.set_xticks([])
axes1.grid(axis = 'x')

axes2.bar(p_by_completion.index.values, p_by_completion['price'], color = ['salmon'], alpha = 0.8)
axes2.set_xlabel('year_of_completion', fontsize = 18)
axes2.set_ylabel('Price', fontsize = 18)
axes2.set_xticks([])
axes2.grid(axis = 'x')



#3. 파생변수 생성
#3.1 아파트 연차 변수 생성
df['until_trans'] = 2021 - df['year_of_completion']

c_by_until = pd.DataFrame(df['price'].groupby(df['until_trans']).count())
p_by_until = pd.DataFrame(df['price'].groupby(df['until_trans']).mean())
std_error = np.std(p_by_until.price, ddof=1) / np.sqrt(len(p_by_until))

plot = plt.figure(figsize=(20,10))
axes1 = plot.add_subplot(1, 2, 1)
axes2 = plot.add_subplot(1, 2, 2)

axes1.bar(c_by_until.index.values, c_by_until['price'], color = ['salmon'], alpha = 0.8)
axes1.set_xlabel('until_trans', fontsize = 20)
axes1.set_ylabel('Count', fontsize = 20)
axes1.set_xticks([])
axes1.grid(axis = 'x')

axes2.bar(p_by_until.index.values, p_by_until['price'], color = ['salmon'], alpha = 0.8, yerr = std_error)
axes2.set_xlabel('until_trans', fontsize = 20)
axes2.set_ylabel('Price', fontsize = 20)
axes2.set_xticks([])
axes2.grid(axis = 'x')


#3.2 거래 일 변수 변환
np.where(df['transaction_date']=='1~10','0',df['transaction_date'])
np.where(df['transaction_date']=='11~20','0',df['transaction_date'])
np.where(df['transaction_date']=='21~28', '2', df['transaction_date'])
np.where(df['transaction_date']=='21~29', '2', df['transaction_date'])
np.where(df['transaction_date']=='21~30', '2', df['transaction_date'])
np.where(df['transaction_date']=='21~31', '2', df['transaction_date'])


#3.3 월 변수와 10일 단위 변수 통합
df['transaction_month_date'] = df['transaction_month']*10 + df['transaction_date']
df[['transaction_month_date','transaction_month','transaction_date']]


#3.4 연도별 주기성을 위한 파생변수 생성(삼각변환)
import math
df['sin_date'] = np.sin(2*np.pi * df.transaction_month_date / 4)
df['cos_date'] = np.cos(2 * np.pi * df.transaction_month_date / 4)

df.drop(['year_of_completion', 'transaction_year_month','transaction_date',' transation_month'], axis=1, inplace=True)



#4. 텍스트 데이터 다루기
#4.1 아파트 이름 한글부분만 뽑기
import re
no_num = re.compile('[^0-9]')

removed = [''.join(filter(str.isalnum,apts)) for apts in df['apt']]
df['apt'] = [''.join(no_num.findall(apts)) for apts in removed]

data.head(5)


#4.2 아파트 이름 DTM
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df = 5).fit_transform(apts.apt)
cv

#4.3 아파트 이름 tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv = TfidfVectorizer(min_df = 5).fit_transform(apts.apt)
tfidfv




#5. 인코딩
data = data.drop(columns =['addr.kr'])


#5.1 원핫 인코딩
dong = pd.get_dummies(data['dong'])
dong.shape

apt = pd.get_dummies(data['apt'])
apt.shape


#5.2 레이블(label) 인코딩 - transaction_year
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['transaction_year'] = le.fit_transform(df['transaction_year'])
df.head()


#5.3 레이블(label) 인코딩
df_label = df
df_label['dong'] = le.fit_transform(df['dong'])
df_label['apt'] = le.fit_transform(df['apt'])
df_label.head()


#5.4 mean encoding
target = 'price'
apt_mean = df.groupby('apt')[target].mean()
df['apt'] = df['apt'].map(apt_mean)
df.head()