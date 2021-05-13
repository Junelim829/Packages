### PSAT
## 클린업 1주차 패키지
# 2019310260 임주은

# 기본세팅
install.packages('plyr')
install.packages('tidyverse')
install.packages('data.table')

library(plyr)
library(tidyverse)
library(data.table)

getwd()
setwd("C:/Users/Jooeun/Desktop/1주차패키지")

data <- fread('data.csv', stringsAsFactors=FALSE, data.table=FALSE)



## Chapter 1. 전처리
#1
str(data)
data %>% is.na %>% colSums
nrow(unique(data))

#2-1
data <- data %>% filter(!is.na(confirmed_date))

#2.2
data <- data %>%
  filter(patient_id != "") %>%
  filter(sex != "") %>% 
  filter(age != "") %>% 
  filter(country != "") %>% 
  filter(province != "") %>% 
  filter(city != "") %>% 
  filter(confirmed_date != "") %>% 
  filter(state != "")

colSums(is.na(data))
nrow(unique(data))

#3
data <- data[data$country == 'Korea',]
data <- subset(data, select= -c(country))

#4
data$province <- data$province %>% 
  revalue(c("서울"="서울특별시",
            "부산"="부산광역시",
            "대구"="대구광역시",
            "인천"="인천광역시",
            "대전"="대전광역시",
            "세종"="세종특별자치시",
            "울산"="울산광역시",
            "제주도"="제주특별자치도"))

data %>% head

#5
data$confirmed_date <- data$confirmed_date %>% as.Date
str(data$confirmed_date)

#6 
A <- data %>% group_by(confirmed_date) %>% summarize(n=n())
colnames(A) = c('confirmed_date', 'confirmed_number')
data <- left_join(data, A, by='confirmed_date')
head(data)

#7
data$wday <- format(data$confirmed_date, format='%a')
data$wday <- factor(data$wday, levels=c('일','월','화','수','목','금','토'))
data$wday <- ifelse(data$wday %in% c('일','토', '주말','주중'))
head(data)

#8
tapply(data$confirmed_number, data$age, summary)

## Chapter 2. 시각화
#1-1
data %>% ggplot(aes(x=confirmed_date, y=confirmed_number))+geom_line(colors='lightblue') + ggtitle('코로나 확진자수 추이 \n - 국내인 기준') + theme(plot.title=element_text(hjust=0.5))

#1-2
data %>% group_by(province %>% summarize(n=n()))

#2
B = data %>% group_by(province) %>% summarize(n=n())
data %>% left_join(B, by='province') %>% ggplot(aes(x=reorder(province, n))) 
+ geom_bar(aes(fill=state, color=state), alpha = 0.2, position='stack')+labs(x='지역',y='확진자수', fill='state') + coord_flip()

#3-1
data %>%
  group_by(age, confirmed_date) %>%
  summarise(
    count=n()
  ) %>%
  ggplot(aes(age, count)) +
  geom_boxplot(aes(x=age, y=count, fill=age, colour=age),
               alpha=0.4,
               outlier.shape = NA) + 
  stat_boxplot(geom='errorbar', aes(color = age)) +
  theme_classic() +
  labs(y="일단위 확진자수")


#3-2 
data %>% 
  group_by(age, confirmed_date) %>%
  summarise(
    count=n()
  ) %>% 
  aov(count ~ age, data=.) %>% 
  summary

#Chapter 3.모델링_회귀분석
#0
library(MASS)
library(corrplot)
library(caret)
library(MLmetrics)

#1
Boston <- as.data.frame(MASS::Boston)
B_cor <- cor(Boston)
round(B_cor, 2)
corrplot(B_cor, method='number',type='upper')

#2
Boston %>%
  gather(-medv, key="variable", value = "val") %>%
  ggplot(aes(x=val, y=medv)) + 
  geom_point(aes(x=val, y=medv)) + 
  geom_smooth(method = lm, color ='lightblue') +
  facet_wrap(~variable, nrow=4, scales = "free") +
  labs(title="Scatter plot of dependent variables vs Median Value (medv)")

#3-1
set.seed(1234)
train_index <- createDataPartition(Boston$medv, p=0.7, list=FALSE)
train_data <- Boston[train_index,]
test_data <- Boston[-train_index,]

#3-2
Fit <- train_data %>% 
  train(medv ~ ., data=., method="lm")
summary(Fit)
medv_pred <- predict(Fit, newdata = test_data)
medv_true = test_data$medv
RMSE(medv_pred, medv_true)

#3-3
print('RMSE , 즉 평균 제곱근 오차 (Root Mean Square Error)는 추정 값
또는 모델이 예측한 값과 실제 환경에서 관찰되는 값의 차이를 다룰 때
흔히 사용되는 측도이다. 이는 MSE에 루트를 씌운 값이므로 가설을
세운 뒤에 평균 제곱근 오차를 판단하여 조금씩 변화를 주고,
이 변화가 긍정적이면 오차를 최소로 만들 수 있도록 과정을 반복한다.
그러므로, 모델의 RMSE를 낮추려면 예측한 값과 실제 값의 차이를
낮출 수 있도록 평균 제곱근 오차를 최소화 만들면 된다.')


#4
result <- summary(Fit)$coefficients %>% as.data.frame

col <- ifelse(result$Estimate >5, 'red', ifelse(result$Estimate < -2, 'blue', 'yellow'))

result %>%
  ggplot(aes(x=Estimate, y=reorder(rownames(.), Estimate))) +
  geom_col(fill = col, color=col, alpha=0.3) +
  geom_text(aes(label=round(Estimate,2)), position = position_stack(0.5)) +
  theme_classic() +
  theme(legend.position = "none") +
  labs(x="value", y="intercept and variables")