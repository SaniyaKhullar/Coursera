# Coursera
Weight Lifting Exercise
---
title: "Weight Lifting Exercise"
author: "Saniya Khullar"
date: "August 14, 2017"
output:
  html_document: default
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## R Markdown
Overview of Weight-Lifting Exercise:

Six participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: 
Class A:  exactly according to the specification (specified and correct execution of the exercise)
Classes B, C, D, and E refer to common mistakes made while trying to do the dumbbell exercise:
Class B: throwing the elbows to the front 
Class C: lifting the dumbbell only halfway
Class D: lowering the dumbbell only halfway 
Class E: throwing the hips to the front.

The data was collected from accelerometers on the belt, forearm, arm, and dumbbell of the participants, and complied into datasets with 160 features.

The goal of this report is to train a model to predict the manner (class) in which they did the exercise. This model will then be used to predict 20 different test cases (testing data set).

In this situation, I will be building 2 different models, 1 tree and 1 random forest tree. The random forest is an improvement over the traditional 1-tree method because it randomly subsets the predictors (at each decision node of the tree) and aggregates results by combining multiple trees.

I have partitioned my original data into a training and a testing set to help guide my model. 

Load required data:

```{r}
#The training data for this project are available here:
#https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
#The test data are available here:
#https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
# The data was downloaded to a local folder on my computer:


training <- read.csv("C:\\Users\\Saniya.Khullar\\Downloads\\pml-training.csv", header = T)
testing <- read.csv("C:\\Users\\Saniya.Khullar\\Downloads\\pml-testing.csv", header = T)

dim(training)
dim(testing)
```

Installing and loading packages that are necessary here. 

```{r}
#install.packages("caret")
#install.packages("ISLR")
#install.packages("e1071")
#install.packages("party")
#install.packages("ggplot2")
#install.packages("grid")
#install.packages("knitr")
#install.packages("kernlab")
#library(party)
#library(caret)
#library(ISLR)
#library(e1071)
#library(ggplot2)
#library(grid)
#library(knitr)
#library(kernlab)

```

Cleaning and pre-processing the data:

The data has a number of derived features (mean, std. etc.),  that are based on other rows; sometimes there are NA values in the features that lead to errors when the tree and the Random Forest methods are applied.  Therefore, please note that we get rid of any and all rows that have any NA in them...we then see that the data set shrinks from around 19,000 rows to less than 1,000 rows!  
```{r}
# near zero value features


training <- na.omit(training) # getting rid of the NA rows (since those will cause errors when building trees)

# keep numeric and outcome features only
training <- training[, -c(1, 2,3, 4, 5)]
# please note that we get rid of these columns since the row number, user_name, raw_timestamp_part_1, 
# raw_timestamp_part_2, and cvtd_timestamp are not scientific facts and should not really have any influence on the results. 

dim(training) # please view these new dimensions of the training data

```



Training the model

Please partition the training data into training (p = 75%) and validation (the remaining 25%) data sets. 
Only this new training data set will be used to build the tree and randm forest models. 

600 trees are built, using 10-fold cross validation. 10 fold (k = 10) fold cross-validation means that the new training data (new_train) will be divided into 10 groups.  Each time, 9 groups will be used to build a model and 1 unique group (different group for each of the 10 times) will be held-out and used to test the model.  This is cross-validation, and it will be done on this new_train training data set (this 75% of the training data that was chosen for training a model). 
Due to the Cross-Validation, there should be more accurate, less biased results but there would sadly be more variance in the results.  Building 501 trees would take some time (for random forests), but in the end will give good results that are quite accurate.  

In fact, more than the simple tree-method, the Random Forests method will give even stronger, more accurate models that do well even when many predictors are highly correlated with one another.
#set.seed(605)
Please Partition the training data into training and validation data sets:
```{r}
training.rows <- sample(1: nrow(training), 0.75*nrow(training))
#training.rows <- createDataPartition(y = training$classe, p = 0.75,list = FALSE)
new_train <- training[training.rows,]  # the new training data that will be used for model-building is a subset of the original training data (containing about 75% of that data)
new_validation <- training[-training.rows,] # all remaining rows are assigned to the testing (validating) data # part of the original training data. 


#train_cv <- trainControl(method="cv", number = 10)  # 10 since we want 10-fold cross-validation
# train the model
set.seed(605)

```


Building the tree and random forest models:

```{r}
#tree.model <- tree(classe~., data=new_train)

# But in any event, the rf.model, with the random forest approach, would be more reliable since it is aggregating 600 trees and using random subsets of the features at every node to inform its decision.  While it may not be the most interpretable tree, it still will be quite reliable!
```



```{r}
# rf.model <- train(classe~., data=new_train, 
#                  trControl=train_cv, 
#                  method="rf", ntree = 600,
#                  prox = TRUE)


# Model summary

# Below are the summary details of the random forest as well as the confusion matrix and out-of-bag error for # final model. The final model is the tree with the highest accuracy and the lowest out-of-bag error rate.

# print the random forest
# print(rf.model)

# print the final model oob error  (estimated out-of-bag (oob) error)
# rf.model$finalModel$err.rate[501,1]

# print the confusion matrix
# confusionMatrix(rf.model)

# Final Accuracy is given by:
# rf.model$results$Accuracy[2]

# Features that were selected as being the most predictive are:
# rf.model$bestTune$mtry



```


Analysis: 

Below are plots of the error rate by the number of trees used in the models.  There is also a plot of the most important features in the final model. 

```{r}
# plot(rf.model$finalModel)
# varImpPlot(rf.model$finalModel) 

```
Validation
When the remaining 25% of the original training data (the validation set) is used as a 'testing set' to generate a rough idea of the accuracy of the prediction model we have, please note that we get these results, as given by this confusion matrix: 

```{r}
# pred_val_data <- predict(rf.model,newdata=new_validation)
# confusionMatrix(pred_val_data,validate_data$classe)

##           Reference

## Prediction    A    B    C    D    E

##          A 28.4  0.1  0.0  0.0  0.0

##          B  0.0 19.2  0.1  0.0  0.0

##          C  0.0  0.1 17.3  0.2  0.0

##          D  0.0  0.0  0.1 16.2  0.1

##          E  0.0  0.0  0.0  0.0 18.3

```
The model predicts with an accuracy of 0.9951 with a 95% confidence interval of (0.9927, 0.9969). The error rate (1 - accuracy) on the validation set is 0.0049.


Prediction on test set
Below are the predicted results from the unlabelled test set.

```{r}
# pred_test_data <- predict(rf.model, newdata = test_data)
# pred_test_data

```
Please apply this model on the testing data:

```{r}
# pred_test_data <- predict(rf.model,newdata=testing)
# confusionMatrix(pred_test_data,validate_data$classe)

##           Reference

## Prediction    A    B    C    D    E

##          A 1393    6    0    0    0

##          B    2  943    4    0    0

##          C    0    0  845    4    0

##          D    0    0    6  800    2

##          E    0    0    0    0  899
```
