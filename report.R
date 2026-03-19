library(tidyverse)
library(caret)
library(pROC)
library(randomForest)

set.seed(123)

script_path <- normalizePath(sys.frames()[[1]]$ofile)
script_dir <- dirname(script_path)
setwd(script_dir)

if (!file.exists("telecom.csv")) {
  stop("telecom.csv not found in the same folder as report.R")
}

data <- read.csv("telecom.csv", stringsAsFactors = FALSE)

data$Churn <- factor(data$Churn, levels = c("No","Yes"))

factor_vars <- c(
  "gender","SeniorCitizen","Partner","Dependents",
  "PhoneService","MultipleLines","InternetService",
  "OnlineSecurity","OnlineBackup","DeviceProtection",
  "TechSupport","StreamingTV","StreamingMovies",
  "Contract","PaperlessBilling","PaymentMethod"
)

for (v in factor_vars) {
  data[[v]] <- as.factor(data[[v]])
}

if (any(is.na(data$TotalCharges))) {
  data$TotalCharges[is.na(data$TotalCharges)] <- median(data$TotalCharges, na.rm = TRUE)
}

print(summary(select(data, tenure, MonthlyCharges, TotalCharges)))
print(table(data$Churn))
print(prop.table(table(data$Churn)))
print(round(prop.table(table(data$Contract, data$Churn), 1), 3))

print(
  data %>%
    group_by(Churn) %>%
    summarise(
      mean_monthly = mean(MonthlyCharges),
      mean_tenure = mean(tenure),
      .groups = "drop"
    )
)

ggplot(data, aes(x = Churn)) + geom_bar()
ggsave("plot_churn_distribution.png", width = 6, height = 4)

ggplot(data, aes(x = Contract, fill = Churn)) +
  geom_bar(position = "fill")
ggsave("plot_contract_vs_churn.png", width = 7, height = 4)

ggplot(data, aes(x = MonthlyCharges, fill = Churn)) +
  geom_histogram(bins = 30, alpha = 0.6)
ggsave("plot_monthlycharges_vs_churn.png", width = 7, height = 4)

train_index <- createDataPartition(data$Churn, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

set.seed(123)
model_logit <- train(
  Churn ~ .,
  data = train_data,
  method = "glm",
  family = binomial(),
  metric = "ROC",
  trControl = ctrl
)

rf_grid <- expand.grid(mtry = c(2,4,6,8,10))

set.seed(123)
model_rf <- train(
  Churn ~ .,
  data = train_data,
  method = "rf",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = rf_grid,
  ntree = 300
)

model_compare <- bind_rows(
  model_logit$results %>%
    summarise(Model = "Logistic", ROC = max(ROC)),
  model_rf$results %>%
    summarise(Model = "RandomForest", ROC = max(ROC))
)

print(model_compare)

rf_importance <- varImp(model_rf, scale = FALSE)
print(head(rf_importance$importance[order(-rf_importance$importance$Overall), , drop = FALSE], 10))

png("plot_rf_variable_importance.png", width = 800, height = 600)
plot(rf_importance, top = 10)
dev.off()

prob_logit <- predict(model_logit, test_data, type = "prob")[, "Yes"]
prob_rf    <- predict(model_rf, test_data, type = "prob")[, "Yes"]

pred_logit <- factor(ifelse(prob_logit >= 0.5, "Yes", "No"), levels = c("No","Yes"))
pred_rf    <- factor(ifelse(prob_rf >= 0.5, "Yes", "No"), levels = c("No","Yes"))

cm_logit <- confusionMatrix(pred_logit, test_data$Churn, positive = "Yes")
cm_rf    <- confusionMatrix(pred_rf, test_data$Churn, positive = "Yes")

print(cm_logit)
print(cm_rf)

roc_logit <- roc(test_data$Churn, prob_logit, levels = c("No","Yes"))
roc_rf    <- roc(test_data$Churn, prob_rf, levels = c("No","Yes"))

print(auc(roc_logit))
print(auc(roc_rf))

get_metrics <- function(cm, model_name) {
  data.frame(
    Model = model_name,
    Accuracy = unname(cm$overall["Accuracy"]),
    Precision = unname(cm$byClass["Pos Pred Value"]),
    Recall = unname(cm$byClass["Sensitivity"]),
    F1 = unname(cm$byClass["F1"])
  )
}

metrics_default <- bind_rows(
  get_metrics(cm_logit, "Logistic"),
  get_metrics(cm_rf, "RandomForest")
)

print(metrics_default)

thresholds <- seq(0.2, 0.8, 0.05)

threshold_results <- lapply(thresholds, function(th) {
  pred <- factor(ifelse(prob_rf >= th, "Yes", "No"), levels = c("No","Yes"))
  cm <- confusionMatrix(pred, test_data$Churn, positive = "Yes")
  data.frame(
    Threshold = th,
    Recall = unname(cm$byClass["Sensitivity"]),
    Precision = unname(cm$byClass["Pos Pred Value"]),
    F1 = unname(cm$byClass["F1"])
  )
}) %>% bind_rows()

print(threshold_results)

best <- threshold_results[which.max(threshold_results$F1), ]
print(best)

png("plot_roc_curve.png", width = 800, height = 600)
plot(roc_logit)
plot(roc_rf, add = TRUE, col = "red")
dev.off()