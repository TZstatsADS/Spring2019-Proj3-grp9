#########################################################
### Train a classification model with training features ###
#########################################################

### Author: Chao Yin
### Project 3


train <- function(dat_train, label_train, par=NULL){
  
  ### Train a Gradient Boosting Model (GBM) using processed features from training images
  
  ### Input: 
  ###  -  features from LR images 
  ###  -  responses from HR images
  ### Output: a list for trained models
  
  ### load libraries
  library('gbm')
  library('foreach')
  library('doParallel')
  
  
  ### creat model list
  modelList <- list()
  
  ### Train with gradient boosting model
  if(is.null(par)){
    depth <- 3
    lr <- 0.001
  } else {
    depth <- par$d
    learning_rate <- par$lr
  }
  
  cores <- detectCores(logical=F)
  cl <- makeCluster(cores)
  registerDoParallel(cl, cores=cores)
  clusterEvalQ(cl, library('gbm'))
  
  modelList <- foreach(i=1:12) %dopar%{
    c1 <- (i-1) %% 4 + 1
    c2 <- (i-c1) %/% 4 + 1
    featMat <- dat_train[, , c2]
    labMat <- label_train[, c1, c2]
    fit_gbm <- gbm::gbm.fit(x=featMat, y=labMat,
                       n.trees=200,
                       distribution="gaussian",
                       interaction.depth=depth, 
                       bag.fraction = 0.5,
                       verbose=FALSE,
                       shrinkage = learning_rate)
    best_iter <- gbm::gbm.perf(fit_gbm, method="OOB", plot.it = FALSE)
    list(fit=fit_gbm, iter=best_iter)
  }
  
  stopImplicitCluster()
  stopCluster(cl)
  
  return(modelList)
}
