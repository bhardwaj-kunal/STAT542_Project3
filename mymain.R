library(tidyverse)
library(readr)
library(glmnet)
library(pROC)

train <- read_csv("train.csv", col_types = cols(sentiment = col_number()))
test <- read_csv("test.csv")

x_train <- as.matrix(train %>% select(-c(id, review, sentiment)))
y_train <- train$sentiment
x_test <- as.matrix(test %>% select(-c(id, review)))

# Fit Elastic Net with Cross-Validation
alpha_values <- 0.6
auc_results <- list()

start_time <- Sys.time()
set.seed(542)
for (alpha in alpha_values) {
  cv_fit <- cv.glmnet(
    x = x_train,
    y = y_train,
    alpha = alpha,
    family = "binomial",
    standardize = FALSE,
    type.measure = "auc"
  )
  
  best_auc <- max(cv_fit$cvm)
  best_lambda <- cv_fit$lambda.min
  
  auc_results[[as.character(alpha)]] <- list(
    alpha = alpha,
    best_auc = best_auc,
    best_lambda = best_lambda,
    model = cv_fit
  )
}

end_time <- Sys.time()
elapsed_time <- end_time - start_time
print(elapsed_time)

# Find the best model
best_alpha <- names(auc_results)[which.max(sapply(auc_results, function(x) x$best_auc))]
best_model <- auc_results[[best_alpha]]$model
best_lambda <- auc_results[[best_alpha]]$best_lambda

predictions <- predict(best_model, s = best_lambda, newx = as.matrix(x_test), type = "response")

cat("Best alpha:", best_alpha, "\n")
cat("Best lambda:", best_lambda, "\n")

write.csv(setNames(data.frame(test$id, predictions), c("id", "prob")), "mysubmission.csv", row.names = FALSE)


