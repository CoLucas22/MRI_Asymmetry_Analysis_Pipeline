library(dplyr)
library(pROC)
library(ggplot2)
library(tidyr)


train_prediction <- function(train_dataset, validation_dataset, verbose=FALSE){
    #train logistic regression model
    model <- glm(label ~ ., data = train_dataset[, -1], family = binomial)

    #AUC on the training set
    train_predictions <- predict(model, type = "response")
    train_labels <- train_dataset$label
    train_roc <- roc(train_labels, train_predictions)
    train_auc <- auc(train_roc)
    if(verbose) {
        print(paste("AUC for training set:", train_auc))
    }

    #AUC on the validation set
    validation_predictions <- predict(model, newdata = validation_dataset[, -1], type = "response")
    validation_labels <- validation_dataset$label
    validation_roc <- roc(validation_labels, validation_predictions)
    validation_auc <- auc(validation_roc)
    if (verbose) {
    print(paste("AUC for validation set:", validation_auc))
    }
    return(list(model, train_auc, validation_auc, validation_predictions,validation_labels))
}
