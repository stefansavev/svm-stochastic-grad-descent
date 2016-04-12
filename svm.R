rm(list = ls())

#' Read the data from a directory
#'
#' @param dir A directory represented as a string
#' @parem intercept A boolean, whether we want a column of 1's added to the features
#' @return The training and test set
#' @examples
#' data <- readData("/data/example", intercept = FALSE)
#' data$train$features
#' data$train$targets
#' data$test$features
#' data$test$targets
readData <- function(dir, intercept = FALSE) {
  features <-
    0.0 + read.csv(file.path(dir, "features.txt"),
                   sep = ",",
                   header = FALSE)
  features <- as.matrix(features)
  n <- nrow(features)
  if (intercept) {
    features <- cbind(features, matrix(rep(1, n), nrow = n))
  }
  targets <- read.csv(file.path(dir, "target.txt"), header = FALSE)
  targets <- targets[[1]]
  #last 1000 objects are test set
  test.ids <- (n - 1000):n 
  list(
    train = list(features = features[-test.ids,], targets = targets[-test.ids]),
    test  = list(features = features[test.ids,], targets = targets[test.ids])
  )
}

predictScores <- function(X, beta) {
  X %*% matrix(beta, nrow = ncol(X))
}

predictSVM <- function(X, beta) {
  pred.scores <- predictScores(X, beta)
  ifelse(pred.scores > 0, 1.0,-1.0)
}

lossGradient <- function(targets, X, beta) {
  pred.scores <- predictScores(X, beta)
  ifelse(targets * pred.scores > 1, 0,-targets)
}

gradient <- function(loss.grad, beta, X) {
  #beta is not used because this is a linear model
  n = nrow(X)
  m = ncol(X)
  #stack the loss.grad vector m times as column vectors
  loss.grad.mat <- matrix(rep(loss.grad, m), byrow = FALSE, nrow = n)
  #a matrix of 1's to help compute column sums
  ones <- matrix(rep(1, n), nrow = 1) 
  grad <- ones %*% (X * loss.grad.mat)
  #grad <- grad / n
  grad[1, ] #return a vector, not a matrix
}

errorRate <- function(targets, pred.targets) {
  num.errors <- length(which(targets != pred.targets))
  100.0 * num.errors / length(targets)
}

getBatch <- function(n, k) {
  #make sure each batch has the same number of ids
  m <- n - n %% k 
  if (m < 0) {
    #if k is too big, return one batch
    matrix(1:n, byrow = TRUE, nrow = 1)
  }
  else{
    s <- sample(1:n, m, replace = FALSE)
    matrix(s, byrow = TRUE, ncol = k)
  }
}

L2norm <- function(x) {
  sqrt(sum(x * x))
}

#' Compute the coefficients of linear SVM
#'
#' @param train.data A list of targets and features representing the training set
#' @param test.data A list of targets and features representing the test set
#' @param lam. A double. The L2 penalty
#' @param nu. A double. The learning set size
#' @param batch.size. A int. The mini-batch size of gradient descent
#' @param init.beta. A vector. A set of coefficients to enable continued training
#' @return A list containing fitted coefficients, a predict function and train/test errors
#' @examples
#' data <- readData("/data/example", intercept = FALSE)
#' svm.fit <- fit(data$train, data$test, lam = 1.0, nu =  0.0001, batch.size = 1)
#' #support continued training
#' svm.fit <- fit(data$train, data$test, lam = 0.01, nu = 0.001, batch.size = 1, svm.fit$coef)
#' plot(svm.fit$errors.train)
#' plot(svm.fit$errors.test)
#' pred <- svm.fit$predict(data$train$features)
fit <- function(train.data, test.data, lam, nu, batch.size = 1, init.beta = NULL){
  #we use min.iter in the stopping condition, less than 20 is not reliable
  min.iter = 20 
  max.iter = 1000
  
  features <- train.data$features
  targets <- train.data$targets
  batch.size <- max(1, min(nrow(features), batch.size))
  if (is.null(init.beta)){
    beta <- rnorm(ncol(features))
    beta <- beta/L2norm(beta)
  }
  else{
    beta = init.beta
  }
  n = nrow(features)
  #typically lambda is for the sum of losses over all examples
  #need to adjust for the mini-batch case
  norm.lam <- batch.size*lam/n
  l2.grad <- rep(norm.lam, ncol(features)) 
  
  best.beta <- beta
  best.error <- 100.0
  
  errors.train <- rep(100.0, max.iter)
  errors.test <- rep(100.0, max.iter)
  
  for(i in seq(1, max.iter)){
    grad.norm <- L2norm(lossGradient(targets, features, beta))
    err.rate.train <- errorRate(targets, predictSVM(features, beta))
    err.rate.test <- errorRate(test.data$targets, predictSVM(test.data$features, beta))
    rel.grad.norm <- nu*grad.norm / L2norm(beta)
    print(paste("Batch size", batch.size, 
                "Epoch", i, 
                "Train error", err.rate.train, 
                "Test error", err.rate.test, 
                "Gradient norm per example", grad.norm/n, 
                "Relative gradient norm", rel.grad.norm))
    errors.train[i] <- err.rate.train
    errors.test[i] <- err.rate.test
    if (err.rate.train < best.error){
      best.error <- err.rate.train
      best.beta <- beta
    }
    if (i > min.iter){
      iter.lim <- min.iter/2
      m1 <- min(errors.test[1:(i - iter.lim - 1)])
      m2 <- min(errors.test[(i - iter.lim):i])
      print(paste("Test set errors (prev epochs, last few epochs):", m1, m2))
      if (m2 > m1){
        errors.train <- errors.train[1:i]
        errors.test <- errors.test[1:i]
        break
      }
    }
    batches <- getBatch(nrow(features), batch.size)
    
    for(j in 1:nrow(batches)){
      b <- batches[j, ]
      features.b <-  features[b,]
      if (is.null(dim(features.b))){
        dim(features.b) <- c(1, ncol(features))
      }
      targets.b <- targets[b]
      loss.grad.b <- lossGradient(targets.b, features.b, beta)
      grad.beta <- gradient(loss.grad.b, beta, features.b)
      beta <- beta - nu*(grad.beta + l2.grad)
    }
  }  
  list(coef = best.beta, 
       errors.train = errors.train,
       errors.test = errors.test,
       predict = function(data){predictSVM(data, best.beta)})
}

experiment <- function(dir, seeds, batch.sizes){
  data <- readData(dir, intercept = FALSE)
  results <- list()
  for(b in batch.sizes){
    for(seed in seeds){
      set.seed(seed) #affects initial solution and ids within batches
      t <- system.time({
        svm.fit <<- fit(data$train, data$test, lam = 1.0, nu = 0.0001, batch.size = b)
      })
      results[[paste("b=", b, ", seed=",seed, sep="")]] <- list(fit = svm.fit, time = t)
    }
  }
  results
}

dir <- "D:/tmp2/toyproblem/"
output.dir <- "D:/tmp2"
results.v1 <- experiment(dir, c(168686), c(1, 10, 100))
saveRDS(results.v1, file.path(output.dir, "results.v1.rds"))

results.v2 <- experiment(dir, c(168686, 4458166, 67671117, 86816),c(1, 10, 100, 1000, 5000))
saveRDS(results.v2, file.path(output.dir, "results.v2.rds"))

sanityCheck <- function(dir){
  data <- readData(dir, intercept = FALSE)
  targets <- data$train$targets
  features <- data$train$features
  library(penalized)
  pen.fit <- penalized(ifelse(targets > 0, 1, 0), penalize = features, lambda1 = 0, lambda2 = 1, model = "logistic", maxiter = 20)
  
  targets <- data$test$targets
  features <- data$test$features  
  pred.targets.logreg <- predict(pen.fit, penalized=features)
  er <- errorRate(targets, ifelse(pred.targets.logreg > 0.5, 1.0, -1.0))
  print(paste("Error rate for sanity check", er))
}

sanityCheck(dir)

