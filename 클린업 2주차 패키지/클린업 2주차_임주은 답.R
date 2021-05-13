
library(tidyverse)
library(data.table)
library(VIM)


#Chapter 1
#0.
getwd()
setwd('C:/Users/Jooeun/Desktop/2주차패키지')
rawdata <- fread('data.csv')
data <- rawdata

#1.
data <- data %>% select(colnames(data), -ends_with('2'))

#2.
VIM::aggr(data, prop=F, numbers=T, col=c('lightyellow','pink'))

#3-1.
data <- lapply(data, function(x)){
    if(is.numeric(x)){
        x[is.na(x)]=mean(x, na.rm=T)
    }
    return(x)} %>% as.data.frame()
head(data)

#3-2.
na_mode <- function(x){
    unique_x <- unique(x)
    mode <- unique_x[which.max(tabulate(mat(x, unique_x)))]
    mode
    }
data$ownerChange[is.na(data$ownerChange)] <- na_mode(data$ownerChange[!is.na(data$ownerChange)])
head(data)

#4. 
data$OC <- c(ifelse(data$OC == 'open',1,0)) %>% as.factor()
head(data)

#5. 
data = lapply(data, function(x){
    if (bit64::is.integer64(x))
    x = as.numeric(x)
    return(x)
}) %>% as.data.frame()
str(data)



#Chapter 2
library(caret)
library(MLmetrics)
library(randomForest)

#1.
set.seed(1234)
valid <- createDataPartition(data$OC, times=1, p=0.3, list=FALSE)
validation_data <- data[valid,]
train_data <- data[-valid,]
head(train_data)

#2. 
model1 = glm(OC ~ ., family=binomial, data=train_data)
p1 = predict(model1, validation_data)
y1 = ifelse(p1 >= 0.5, 1, 0)
Accuracy(y1, validation_data$OC)

#3.
s_result = step(glm(OC~., family=binomial, data=train_data), direction='both', trace=FALSE)
dim(train_data)
s_result$model %>% dim()

s_var = names(s_result$model)
model2 = glm(OC~., family=binomial, data = train_data[,s_var])
p2 = predict(model2, validation_data[,s_var])
y2 = ifelse(p2 >= 0.5, 1,0)
Accuracy(y2, validation_data$OC)


#4.
acc_rf = expand.grid(mtry=c(3:5), acc=NA)
acc_rf

#5. 
set.seed(1234)
cv = createFolds(train_data$OC, k=5)

for (i in 1:nrow(acc_rf)){
    temp_acc = NULL

    for (j in 1:5){
        valid_idx= c[[j]]
        cv_valid = train_data[valid_idx, s_var]
        cv_train = train_data[-valid_idx, s_var]

        set.seed(1234)
        rfmodel = randomForest(OC~., cv_train,
        mtry= acc_rf[i, 'mtry'], ntree=10)
        rf_pred = predict(rfmodel, newdata=cv_valid)
        temp_acc[j] = Accuracy(rf_pred, cv_valid$OC)
    }
    acc_rf[i, 'acc'] = mean(temp_acc)
}


#6.
acc_rf = acc_rf %>% arrange(desc(acc))

best = acc_rf[1,]
best

#7. 
set.seed(1234)
rf_model = randomForest(OC~., train_data[,s_var],
                        mtry = best$mtry, ntree = 10)

y = predict(rf_model,newdata = validation_data[,s_var])
Accuracy(y,validation_data$OC)


#Chapter 3
#1. 
library(MASS)
set.seed(1234)
test_idx = createDataPartition(Boston$medv, p=0.2, list=FALSE)
train = Boston[-test_idx,]
test = Boston[test_idx,]

#2. 
rf_RMSE = expand.grid(mtry = c(3,4,5), ntree= c(10,100,200), RMSE=NA)

#3. 
set.seed(1234)
cv = createFolds(train$medv, k=5)
for (i in 1:nrow(rf_RMSE)){
    temp_RMSE = NULL

    for (j in 1:5){
        valid_idx = cv[[j]]
        cv_valid = train[valid_idx,]
        cv_train = train[-valid_idx]

    set.seed(1234)
    rf_model = randomForest(medv~., cv_train, mtry=rf_RMSE[i, 'mtry'], ntree=rf_RMSE[i, 'ntree'])
    rf_pred = predict(rf_model, newdata= cv_valid)
    temp_RMSE[j] = RMSE(rf_pred, cv_valid$medv)
    }
    rf_RMSE[i, 'RMSE'] = mean(temp_RMSE)
}

#4. 
rf_RMSE = rf_RMSE %>% arrange(RMSE)
best = rf_RMSE[1,]
best

#5. 
set.seed(1234)
rf_model = randomForest(medv~., data = train, mtry=best$mtry, ntree=best$ntree)
y = predict(rf_model, test)

RMSE(y, test$medv)


