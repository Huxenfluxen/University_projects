#Bodyfat men

### Import and view data
bfm <- read.csv("bodyfatmen.csv")
View(bfm)
names(bfm)
#Fit to a linear model
responsevar <- "density"
predictors <- setdiff(names(bfm), responsevar)
formula_string <- paste(responsevar, "~", paste(predictors, collapse = " + "))
fullmodel <- lm(as.formula(formula_string), data = bfm)


summary(fullmodel)


par(mfrow = c(2, 2)) # Divideing the plot window into 1 by 1
plot(fullmodel)

par(mfrow = c(1, 1))
#QQ-plot sorts the residuals from small to large and plots it. We want normal distribution.
qqnorm(rstudent(fullmodel))  # rstudent = externally studentized
qqline(rstudent(fullmodel))

# (b) Residuals vs. fitted values 
plot(fullmodel$fitted.values, fullmodel$residuals)# horizontal band. Good!

# (c) Residuals vs. regressor variables
# Get the names of predictors used in the model
#predictors <- names(fullmodel$model)[-1]  # Excludes the first column, typically the response variable
# Loop through each predictor and create a plot
# We are looking for any non-linear relation in order to find any non-constant variances
par(mfrow = c(3, 2))
plot(bfm$age, fullmodel$residuals)
plot(bfm$weight, fullmodel$residuals)
plot(bfm$height, fullmodel$residuals) #A little funnel
plot(bfm$neck, fullmodel$residuals) #Funnel
plot(bfm$chest, fullmodel$residuals)
plot(bfm$abdomen, fullmodel$residuals) #Quite curved
par(mfrow = c(3, 3))
plot(bfm$hip, fullmodel$residuals)
plot(bfm$thigh, fullmodel$residuals)
plot(bfm$knee, fullmodel$residuals)
plot(bfm$ankle, fullmodel$residuals)# Funnel
plot(bfm$biceps, fullmodel$residuals)
plot(bfm$forearm, fullmodel$residuals)# Funnel
plot(bfm$wrist, fullmodel$residuals)#Funnel

# (d) Partial residual plots
library(car)  # avPlots
par(mfrow = c(1, 1))
avPlots(fullmodel)
### We observe linear relationships between the y residuals and the x_i
### residuals, which confirms that the regression variables belong in the
### model.


# (e)
library(MASS)  # studres
par(mfrow = c(1, 1))
#boxcox(fullmodel)
par(mfrow = c(1, 2))
\plot(studres(fullmodel), xlab="x", ylab="Studentized residual")
plot(rstudent(fullmodel), xlab="x", ylab="R-Student residual")
### With these plots we can identify influential points and potential
### outliers: All points with |error| > 2 can be outliers here.

#We can use Cook's distance to be more certain
cooksD <- cooks.distance(fullmodel)
# Identify influential observations
n <- length(cooksD)
k <- length(fullmodel$coefficients) - 1
threshold <- 4 / (n - k - 1)
par(mfrow = c(1, 1))
plot(cooksD, type="h", main="Cook's Distance", ylab="Cook's Distance", xlab="Observation Index")
abline(h=threshold, col="red", lty=2)  # Add threshold line

#Here we check both which that are above 2 deviations of the mean as well as the
# if they are also influential according to Cook's distance
rstudent_res <- rstudent(fullmodel)
outliers <- which(abs(rstudent_res) > 2)
influential_obs <- which(cooksD > threshold)
out_and_influential <- intersect(outliers, influential_obs)
#Check them (analyse)
#cooksD[influential_obs] #39 is very influential on almost all points increasing
# the expectancy too much, also 83 have very large ankles which might be caused
# by some condition, hence it is legitimate to remove 83 as well

### DFFITS
# Does not say much else than cook and the other influential tests
dffitscutoff <- 2 * sqrt((k-1) / n)  # MPV p. 219
fullmodeldffits <- dffits(fullmodel)
fullmodeldffits[abs(fullmodeldffits) > dffitscutoff]

# Built-in functions
#par(mfrow = c(2, 3))
#plot(solar.model, 1:5)

#Rigorous influential checking including cook and QQ plot
influenceModel <- influence.measures(fullmodel)
summary(influenceModel)
#Do some DFBETAS analysis
par(mfrow = c(2,3))
plot(fullmodel, 1:5)

# remove the influential observations that is also signifacnt outliers
# We remove point 39 since it is the most influential and outlier
bfmClean <- bfm[-39, ]
clean_model <- lm(formula_string, data = bfmClean)
summary(clean_model)
View(bfmClean)

par(mfrow = c(2, 2))
plot(clean_model)

#Now we can perform a regressor transformation to handle the non-linear 
#relationship of the regressors/predictors.
Trans <- boxTidwell(density ~ chest + wrist, data=bfmClean) # rather unreliable
chestT <- Trans$result[1][1]
wristT <- Trans$result[2][1]

clean_modelTrans <- lm(density ~ age + weight + height + neck + I(chest^chestT) + abdomen + hip + thigh + knee + ankle + biceps + forearm + wrist, data = bfmClean)
summaryClean = summary(clean_modelTrans)
par(mfrow = c(2, 2)) # Dividing the plot window into 1 by 1
plot(clean_modelTrans)

par(mfrow = c(1, 1))
#QQ-plot sorts the residuals from small to large and plots it. We want normal distribution.
qqnorm(rstudent(clean_modelTrans))  # rstudent = externally studentized
qqline(rstudent(clean_modelTrans))

# (b) Residuals vs. fitted values 
plot(clean_modelTrans$fitted.values, clean_modelTrans$residuals)# horizontal band. Good!

# (c) Residuals vs. regressor variables
# Get the names of predictors used in the model
#predictors <- names(fullmodel$model)[-1]  # Excludes the first column, typically the response variable
# Loop through each predictor and create a plot
# We are looking for any non-linear relation in order to find any non-constant variances
par(mfrow = c(1, 1))
plot(bfmClean$age, clean_modelTrans$residuals)
plot(bfmClean$weight, clean_modelTrans$residuals)
plot(bfmClean$height, clean_modelTrans$residuals) #A little funnel
plot(bfmClean$neck, clean_modelTrans$residuals) #Funnel
plot(bfmClean$chest, clean_modelTrans$residuals)
plot(bfmClean$abdomen, clean_modelTrans$residuals) #Quite curved
par(mfrow = c(3, 3))
plot(bfmClean$hip, clean_modelTrans$residuals)
plot(bfmClean$thigh, clean_modelTrans$residuals)
plot(bfmClean$knee, clean_modelTrans$residuals)
plot(bfmClean$ankle, clean_modelTrans$residuals)# Funnel
plot(bfmClean$biceps, clean_modelTrans$residuals)
plot(bfmClean$forearm, clean_modelTrans$residuals)# Funnel
plot(bfmClean$wrist, clean_modelTrans$residuals)#Funnel

# (d) Partial residual plots
library(car)  # avPlots
par(mfrow = c(1, 1))
avPlots(clean_modelTrans)
### We observe linear relationships between the y residuals and the x_i
### residuals, which confirms that the regression variables belong in the
### model.


# (e)
library(MASS)  # studres
par(mfrow = c(1, 1))
#boxcox(fullmodel)
par(mfrow = c(1, 2))
plot(studres(clean_modelTrans), xlab="x", ylab="Studentized residual")
plot(rstudent(clean_modelTrans), xlab="x", ylab="R-Student residual")
#Now a thorough analysis has been made

###MULTICOLLINEARITY###

# (a) Correlation matrix to detect multicollinearity
X <- model.matrix(clean_modelTrans)
XtX <- cor(X[,-1])  # X^T X in correlation form

# Using other built-in functions
library("GGally")  # ggpairs
ggpairs(data = data.frame(bfmClean$age, bfmClean$weight, bfmClean$height, bfmClean$neck, bfmClean$chest, bfmClean$abdomen, bfmClean$hip, bfmClean$thigh, bfmClean$knee, bfmClean$ankle, bfmClean$biceps, bfmClean$forearm, bfmClean$wrist))
#Very high correlation! We need to get around collinearity!


# (b) Variance inflation factor and condition number
library(car)  # vif
vif(clean_modelTrans)  # cutoff value is 10 (MPV p. 118)
solve(XtX)  # = (X^T X)^{-1}. Compare diagonal values with VIFs

### Condition number, cutoff value is 388.3738
bfmClean.eigen <- eigen(XtX)
max(bfmClean.eigen$values) / min(bfmClean.eigen$values)




### Ridge regression and LASSo ###

# Split data into training and test sets
set.seed(689)
n <- nrow(bfmClean)
train <- sample(1:n, n / 2) # split in half
for (i in 2:ncol(bfmClean)) {  # normalize a bit?
  bfmClean[,i] <- scale(bfmClean[,i])
}
bfmClean.train <- bfmClean[train, ]
bfmClean.test <- bfmClean[-train, ]

bf.train.model <- lm(density ~ ., data = bfmClean.train)

model.pred <- predict(bf.train.model, bfmClean.test)
model.MSE <- mean((bfmClean.test$density - model.pred) ^ 2)
model.MSE

# (c) Ridge regression
library(glmnet)  # elastic net

# Prepare the matrix of predictors and the response vector for training data
x_train <- as.matrix(bfmClean.train[, -which(names(bfmClean.train) == "density")])
y_train <- bfmClean.train$density

# And for test data
x_test <- as.matrix(bfmClean.test[, -which(names(bfmClean.test) == "density")])
y_test <- bfmClean.test$density

# Fit the ridge regression model
ridge.model <- glmnet(x_train, y_train, alpha = 0)
set.seed(689) # For reproducibility
cv.ridge <- cv.glmnet(x_train, y_train, alpha = 0)

# Plot the cross-validation result to visualize lambda selection
par(mfrow = c(1,1))
plot(cv.ridge)

ridge.pred <- predict(ridge.model, s = cv.ridge$lambda.min, newx = x_test)

# Calculate the MSE for the ridge regression model
ridge.MSE <- mean((y_test - ridge.pred) ^ 2)
ridge.MSE

# (d) Lasso
bfmClean.lasso <- cv.glmnet(x_train, y_train, alpha=1)  # alpha = 1 means lasso
bfmClean.lasso$lambda.1se
plot(bfmClean.lasso)

### Report test error
lasso.pred <- predict(bfmClean.lasso, s = bfmClean.lasso$lambda.1se, newx = x_test)
lasso.MSE <- mean((bfmClean.test$density - lasso.pred)^2)
lasso.MSE

### Report coefficient estimates
coefs.lasso <- coef(bfmClean.lasso)  # coefficients for lambda.1se
coefs.lasso[rowSums(coefs.lasso) != 0,]  # nonzero coefficients

# (e) Principal component regression
library(pls)  # pcr
# Automatically select the number of components with cross-validation
pcr.model <- pcr(density ~ ., data = bfmClean.train, scale = TRUE, validation = "CV")

# Summary of the PCR model to check how many components were selected
summary(pcr.model)


# Assuming you choose a specific number of components, let's say chosen.components
chosen.components <- which.min(pcr.model$validation$PRESS)

# Make predictions on the test set
predictions <- predict(pcr.model, bfmClean.test, ncomp = chosen.components)

# Calculate the MSE
pcr.MSE <- mean((bfmClean.test$density - predictions)^2)
pcr.MSE


# (g) Compare the methods
model.MSE
ridge.MSE
lasso.MSE
pcr.MSE


###STEPWISE REGRESSION###
#Want to solve the multicollinearity problem

# (d) Forward and backward stepwise selection
### Forward
library(leaps)

backward.model <- regsubsets(density ~ ., data = bfmClean.train, nvmax = 13, method="backward")
backward.summary <- summary(backward.model)
bestModel <- which.min(backward.summary$bic)
which.min(backward.summary$bic)
which.max(backward.summary$adjr2)

adjRsquared = backward.summary$adjr2[9]
ccpp = backward.summary$cp[6]
biccen = backward.summary$bic[3]

bestCoeffs <- coef(backward.model, id = bestModel)
selectedVars <- names(bestCoeffs)[-1]
formulaBW <- as.formula(paste("density ~", paste(selectedVars, collapse = " + ")))
lmBest <- lm(formulaBW, data = bfmClean.train)
plot(lmBest)

backmodel.pred <- predict(lmBest, bfmClean.test)
backmodel.MSE <- mean((bfmClean.test$density - backmodel.pred) ^ 2)
backmodel.MSE

summary(lmBest)
# Assuming your full dataset is bodyfat and lm.best is your model fitted on the full dataset or training set

# Prepare the data for cross-validation
# Note: If lm.best was fitted on a subset, you might need to refit it on the full dataset for an accurate cross-validation
# Perform 10-fold cross-validation
genLin_clean_modelTrans <- glm(density ~ age + weight + height + neck + I(chest^chestT) + abdomen + hip + thigh + knee + ankle + biceps + forearm + wrist, data = bfmClean)
cv.result <- cv.glm(bfmClean, genLin_clean_modelTrans, K=20)

# Extract the estimated mean squared error (MSE)
cv.mse <- cv.result$delta[1]
print(cv.mse)

library(caret)
train_control <- trainControl(method = "cv", number = 10, summaryFunction = defaultSummary)

# Example for linear regression
linear_model_cv <- train(density ~ ., data = bfmClean, method = "lm", trControl = train_control)
print(linear_model_cv$results)

# Example for PCR with caret package
pcr_model_cv <- train(density ~ ., data = bfmClean, method = "pcr", trControl = train_control, tuneLength = 20)
print(pcr_model_cv$results)

### Bootstrapping ###

library(boot)
cv.result <- cv.glm(bfmClean, clean_modelTrans, K=10)
print(cv.result$delta[1])  # This prints the estimated cross-validation error.

# Define a statistic function that computes the model's coefficients
boot_statistic <- function(data, indices) {
  sample_data <- data[indices, ]  # Resample with the given indices
  model <- lm(density ~ ., data = sample_data)
  return(coef(model))
}

# Perform bootstrapping
results <- boot(data=bfmClean, statistic=boot_statistic, R=1000)  # R is the number of bootstrap replications
print(results)


