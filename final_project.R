thoracic <- read.csv('thoracic_surgery.csv')

#### random foresting ####
library(randomForest)
library(gbm)

#changing T and F to 1 and 0 respectively
thoracic[thoracic == "TRUE"] <- 1
thoracic[thoracic == "FALSE"] <- 0
write.csv(thoracic, "binary_thoracic.csv", row.names = FALSE)

thoracic$DGN <- as.factor(thoracic$DGN)
thoracic$Risk1Yr <- as.factor(thoracic$Risk1Yr)
thoracic$PRE6 <- as.factor(thoracic$PRE6)
thoracic$PRE7 <- as.factor(thoracic$PRE7)
thoracic$PRE8 <- as.factor(thoracic$PRE8)
thoracic$PRE9 <- as.factor(thoracic$PRE9)
thoracic$PRE10 <- as.factor(thoracic$PRE10)
thoracic$PRE11 <- as.factor(thoracic$PRE11)
thoracic$PRE14 <- as.factor(thoracic$PRE14)
thoracic$PRE17 <- as.factor(thoracic$PRE17)
thoracic$PRE19 <- as.factor(thoracic$PRE19)
thoracic$PRE25 <- as.factor(thoracic$PRE25)
thoracic$PRE30 <- as.factor(thoracic$PRE30)
thoracic$PRE32 <- as.factor(thoracic$PRE32)

set.seed(2024)
index.train <- sample(1:nrow(thoracic), 0.8 * nrow(thoracic))
data.train <- thoracic[index.train,]
data.test <- thoracic[-index.train,]

#bagging
bagging.thoracic <- randomForest(Risk1Yr ~ ., mtry = 13, ntree = 500, importance = TRUE, data = data.train)

#random forest
rf.thoracic <- randomForest(Risk1Yr~., mtry = 4, ntree = 500, data=data.train)

#boosting
boost.thoracic <- gbm(as.character(Risk1Yr)~., n.trees = 500, distribution = "bernoulli", data = data.train)

yhat.test.bag <- predict(bagging.thoracic, newdata = data.test, type = 'response')
yhat.test.rf <- predict(rf.thoracic, newdata = data.test, type = 'response')
yhat.test.prob <- predict(boost.thoracic, newdata = data.test, type = 'response')
yhat.test.boost <- ifelse(yhat.test.prob > 0.5, 1, 0)

mean(yhat.test.bag == data.test$Risk1Yr) 
mean(yhat.test.rf == data.test$Risk1Yr) 
mean(yhat.test.boost == data.test$Risk1Yr) 

####random foresting model####
#confusion matrix
confusion.matrix <- table(yhat.test.rf, data.test$Risk1Yr)
confusion.matrix
#accuracy 
accuracy_rf <- sum(diag(confusion.matrix)) / sum(confusion.matrix)
accuracy_rf
#precision and recall
precision <- sum(diag(confusion.matrix)) / colSums(confusion.matrix)
recall <- diag(confusion.matrix) / rowSums(confusion.matrix)
precision
recall

####bagging and boosting models ####
# confusion matrix for bagging model
confusion_matrix_bag <- table(data.test$Risk1Yr, yhat.test.bag)
# accuracy for bagging model
accuracy_bag <- sum(diag(confusion_matrix_bag)) / sum(confusion_matrix_bag)
# precision for bagging model
precision_bag <- diag(confusion_matrix_bag) / colSums(confusion_matrix_bag)
# recall for bagging model
recall_bag <- diag(confusion_matrix_bag) / rowSums(confusion_matrix_bag)
# confusion matrix for boosting model
confusion_matrix_boost <- table(data.test$Risk1Yr, yhat.test.boost)
#accuracy for boosting model
accuracy_boost <- sum(diag(confusion_matrix_boost)) / sum(confusion_matrix_boost)
# precision for boosting model
precision_boost <- diag(confusion_matrix_boost) / colSums(confusion_matrix_boost)
# recall for boosting model
recall_boost <- diag(confusion_matrix_boost) / rowSums(confusion_matrix_boost)

#ROC curves
library(PRROC)
y.truth <- data.test$Risk1Yr
#random forest
plot(roc.curve(scores.class0 = yhat.test.rf[y.truth==1], 
               scores.class1 = yhat.test.rf[y.truth==0], curve = TRUE),
     ylab='True Postive Rate', xlab='False Negative Rate (1 - True Negative Rate)', main = 'Random Forest Model ROC Curve')
#bagging
plot(roc.curve(scores.class0 = yhat.test.bag[y.truth==1], 
               scores.class1 = yhat.test.bag[y.truth==0], curve = TRUE),
     ylab='True Postive Rate', xlab='False Negative Rate (1 - True Negative Rate)', main = 'Bagging Model ROC Curve')
plot(roc.curve(scores.class0 = yhat.test.boost[y.truth==1], 
               scores.class1 = yhat.test.boost[y.truth==0], curve = TRUE),
     ylab='True Postive Rate', xlab='False Negative Rate (1 - True Negative Rate)', main = 'Boosting Model ROC Curve')

#### decision tree ####
library(tree)
library(rpart.plot)

thoracic <- read.csv("binary_thoracic.csv")

thoracic$DGN <- as.factor(thoracic$DGN)
thoracic$Risk1Yr <- as.factor(thoracic$Risk1Yr)
thoracic$PRE6 <- as.factor(thoracic$PRE6)
thoracic$PRE7 <- as.factor(thoracic$PRE7)
thoracic$PRE8 <- as.factor(thoracic$PRE8)
thoracic$PRE9 <- as.factor(thoracic$PRE9)
thoracic$PRE10 <- as.factor(thoracic$PRE10)
thoracic$PRE11 <- as.factor(thoracic$PRE11)
thoracic$PRE14 <- as.factor(thoracic$PRE14)
thoracic$PRE17 <- as.factor(thoracic$PRE17)
thoracic$PRE19 <- as.factor(thoracic$PRE19)
thoracic$PRE25 <- as.factor(thoracic$PRE25)
thoracic$PRE30 <- as.factor(thoracic$PRE30)
thoracic$PRE32 <- as.factor(thoracic$PRE32)

set.seed(2023)
index.train <- sample(1:nrow(thoracic), 0.8 * nrow(thoracic))
data.train <- thoracic[index.train,]
data.test <- thoracic[-index.train,]

tree.thoracic <-rpart(Risk1Yr~., data = data.train)
rpart.plot(tree.thoracic)

predictions <- predict(tree.thoracic, data.test, type = "class")

# calculate accuracy
accuracy <- mean(predictions == data.test$Risk1Yr)

# another way to plot, not as visually appealing
plot(tree.thoracic)
text(tree.thoracic, use.n = TRUE, all = TRUE, cex = 0.8)

#### KNN classification ####
thoracic <- read.csv('thoracic_surgery.csv')
thoracic[thoracic == "TRUE"] <- 1
thoracic[thoracic == "FALSE"] <- 0
thoracic <- thoracic[, -c(1,4,10)]

set.seed(2022)
index.train <- sample(1:nrow(thoracic), 0.8 * nrow(thoracic))
data.train <- thoracic[index.train,]
data.test <- thoracic[-index.train,]

data.train <- na.omit(data.train)
data.test <- na.omit(data.test)

library(class)
yhat.test.knn <- knn(
  train = data.train[, -14],
  test = data.test[, -14],
  cl = data.train$Risk1Yr,
  k = 7
)

# calculating mean, conf matrix, accuracy, precision, recall
mean(data.test$Risk1Yr==yhat.test.knn)
conf_knn <- table(yhat.test.knn, data.test$Risk1Yr)
accuracy_knn <- sum(diag(conf_knn)) / sum(conf_knn) * 100
precision_knn <- diag(conf_knn) / colSums(conf_knn) * 100
recall_knn <- diag(conf_knn) / rowSums(conf_knn) * 100
y.truth.knn <- data.test$Risk1Yr

#ROC curve
plot(roc.curve(scores.class0 = yhat.test.knn[y.truth==1], 
               scores.class1 = yhat.test.knn[y.truth==0], curve = TRUE),
     ylab='True Postive Rate', xlab='False Negative Rate (1 - True Negative Rate)', main = 'KNN Model ROC Curve')


# all model results into dataframe for easier visualization
model_results <- data.frame(
  Model = c("Random Forest", "Bagging", "Boosting", "KNN"),
  Accuracy = c(90.43, 89.36, 81.91, accuracy_knn),
  Precision_0 = c(98.83, 91.30, 90.59, precision_knn[1]),
  Precision_1 = c(10.63, 0.00, 0.00, precision_knn[2]),
  Recall_0 = c(91.40, 97.67, 89.53, recall_knn[1]),
  Recall_1 = c(0.00, 0.00, 0.00, recall_knn[2]),
  AUROC = c(49.42, 48.84, 44.77, 49.42)
)

# Print the results
cat("Model", "Accuracy", "Precision_0", "Precision_1", "Recall_0", "Recall_1", "AUROC\n")
for (i in 1:nrow(model_results)) {
  cat(model_results$Model[i], model_results$Accuracy[i], 
      model_results$Precision_0[i], model_results$Precision_1[i], 
      model_results$Recall_0[i], model_results$Recall_1[i], 
      model_results$AUROC[i], "\n")
}


