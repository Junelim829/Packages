
#Chapter 1
#0
library(tidyverse)
library(data.table)
library(gridExtra)

setwd('C:/Users/Jooeun/Desktop/3주차패키지')
train=fread('data.csv')
test=fread('test.csv')

#1.
train$bmi <- as.numeric(train$bmi, stringsAsfactors=FALSE)
train$bmi[is.na(train$bmi)] = mean(train$bmi, na.rm=T)

#2.
train = train %>% mutate_if(is.character, as.factor)
str(train)

#3.
train = train %>% select(-id)

#4. 
train1 <- train %>% group_by(stroke, work_type, smoking_status, Residence_type, hypertension, heart_disease, gender, ever_married) %>% summarize(count=c())
train1 <- train %>% gather(key, value, -1)
p1 <- train1 %>% filter(stroke==0) %>% ggplot(aes(x=key, y=value, fill=value))+geom_col()+coord_flip()
p2 <- train1 %>% filter(stroke==1) %>% ggplot(aes(x=key, y=value, fill=value))+geom_col()+coord_flip()
grid.arrange(p1, p2, ncol=2)

#5.
train2 <- train %>% group_by(stroke, age, avg_glucose_level, bmi) %>% summarize(count=c())
train2 <- train2 %>% gather(key, value,-1)
p3 <- train2 %>% filter(stroke==1) %>% ggplot(aes(x= value, fill=key))+geom_density()
p4 <- train2 %>% filter(stroke==0) %>% ggplot(aes(x=value, fill=key))+geom_density()
grid.arrange(p3, p4)

#6. 
cat_var = train %>% select(where(is.factor),-stroke) %>% colnames()
result = data.frame(cat_var,chi=rep(NA,7),stringsAsFactors = F)
for(i in 1:7){
  data_table = table(as_vector(train[,cat_var[i],with=F]),train$stroke)
  a= chisq.test(data_table)
  if(a$p.value < 0.05){
    result$chi[i] = 'denied'
  } else{
    result$chi[i] = 'accept'
  }
}
result

#7.
train = train %>% select(-c(gender,Residence_type))



#Chapter 2
library(catboost)
library(caret)
library(MLmetrics)

#0
print('catboost모델은 오버피팅을 감소시키기 위한 기법으로 대표적인 파라미터로는 손실함수가 있다.')

#1
logloss_cb = expand.grid(depth=c(4,6,8), iterations=c(100,200), logloss=NA)
logloss_cb

#2
set.seed(1234)
idx = createFolds(train$stroke, k=5, list=FALSE)
cv_start = Sys.time()
for (i in 1:6){
    temp= NULL

    for (j in 1:5){
        cv_train = train[which(idx == k), ]
        cv_test = train[which(idx==k), ]

        train_pool = catboost.load_pool(data = cv_train[, ='stroke'], label=as.double(cv_train$stroke))
        test_pool = catboost.load_pool(data=cv_test[,-'stroke'], label= as.double(cv_train$stroke))
    
    start = Sys.time()
    param = list(loss_function = 'Logloss', random_seed = 1234, iterations= logloss_cb[i,2], depth=logloss_cb[i,1], custom_loss='Logloss')

    cat = catboos.train(learn_pool=train_pool, test_pool, params=param)
    test_error = read.table('catboost_info/test_error.tsv', sep='\t', header=TRUE)
    temp = c(temp, test_error[nrow(test_error), 'Logloss'])
    }
    logloss_cb[i, 'logloss'] = min(temp)
}

#3
logloss_cb %>% arrange(logloss) %>% head(1)

#4
train_pool_t = catboost.load_pool(data=train[,-'stroke'], label=as.double(train$stroke))
test_pool_t = catboost.load_pool(data=test[,-'stroke'], label = as.double(test$stroke))

param = list(loss_function='Logloss', random_seed = 1234, iterations=200, depth=8, custom_loss='Logloss')
cat_total = catboost.train(learn_pool= train_pool_t, test_pool=test_pool_t, params=param)



#Chapter 3
library(factoextra)
library(cluster)

#1
age_ = scale(test$age)
avg_ = scale(test$avg_glucose_level)
bmi_ = scale(test$bmi)

A = cbind(age_, avg_, bmi_)


#2
fviz_nbclust(A, kmeans, method='wss')
fviz_nbclust(A, kmeans, method = 'silhouette')

#3
set.seed(1234)
B <- kmeans(A, centers = 3, nstart=1, iter.max=30)
fviz_cluster(B, data=A)

#4
pp = train %>% select(where(is.numeric)) %>% mutate(cluster= B$cluster) 
p7 = pp %>% ggplot(aes(x=factor(cluster),y=age_)) + geom_boxplot(outlier.shape = NA, alpha=0.6,fill=c( '#845ec2', '#ffc75f', '#ff5e78'),color=c( '#845ec2', '#ffc75f', '#ff5e78')) + stat_boxplot(geom='errorbar',color=c( '#845ec2', '#ffc75f', '#ff5e78')) + labs(x='cluster') + theme_bw()
p8 = pp %>% ggplot(aes(x=factor(cluster),y=avg_)) + geom_boxplot(outlier.shape = NA, alpha=0.6,fill=c( '#845ec2', '#ffc75f', '#ff5e78'),color=c( '#845ec2', '#ffc75f', '#ff5e78')) + stat_boxplot(geom='errorbar',color=c( '#845ec2', '#ffc75f', '#ff5e78')) + labs(x='cluster') + theme_bw()
p9 = pp %>% ggplot(aes(x=factor(cluster),y=bmi_)) + geom_boxplot(outlier.shape = NA, alpha=0.6,fill=c( '#845ec2', '#ffc75f', '#ff5e78'),color=c( '#845ec2', '#ffc75f', '#ff5e78')) + stat_boxplot(geom='errorbar',color=c( '#845ec2', '#ffc75f', '#ff5e78')) + labs(x='cluster') + theme_bw()
grid.arrange(p7,p8,p9,ncol=3)
