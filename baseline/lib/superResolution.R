########################
### Super-resolution ###
########################

### Author: Chao Yin
### Project 3

superResolution <- function(LR_dir, HR_dir, modelList){
  
  ### Construct high-resolution images from low-resolution images with trained predictor
  
  ### Input: a path for low-resolution images + a path for high-resolution images 
  ###        + a list for predictors
  
  ### load libraries
  library("EBImage")
  n_files <- length(list.files(LR_dir))
  
  ### read LR/HR image pairs
  for(i in 1:n_files){
    imgLR <- readImage(paste0(LR_dir,  "img", "_", sprintf("%04d", i), ".jpg"))
    pathHR <- paste0(HR_dir,  "img", "_", sprintf("%04d", i), ".jpg")
    featMat <- array(NA, c(dim(imgLR)[1] * dim(imgLR)[2], 8, 3))
    
    height <- dim(imgLR)[1] 
    width <- dim(imgLR)[2]
    n_points <- height * width

    centre <- array(imgLR[abind(rep(1:height, 3*width), rep(rep(1:width, rep(height,width)),3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3))
    img_pad <- abind(array(0, c(height, 1, 3)), imgLR, array(0, c(height, 1, 3)), along = 2)
    img_pad <- abind(array(0, c(1, width + 2, 3)), img_pad, array(0, c(1, width + 2, 3)), along = 1)
    
    pad_row <- 1:height + 1
    pad_col <- 1:width + 1
    
    featMat[ , 1, ] <- array(img_pad[abind(rep(pad_row-1, 3*width), rep(rep(pad_col-1, rep(height,width)),3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    featMat[ , 2, ] <- array(img_pad[abind(rep(pad_row, 3*width), rep(rep(pad_col-1, rep(height,width)),3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    featMat[ , 3, ] <- array(img_pad[abind(rep(pad_row+1, 3*width), rep(rep(pad_col-1, rep(height,width)),3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    featMat[ , 4, ] <- array(img_pad[abind(rep(pad_row-1, 3*width), rep(rep(pad_col, rep(height,width)),3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    featMat[ , 5, ] <- array(img_pad[abind(rep(pad_row+1, 3*width), rep(rep(pad_col, rep(height,width)),3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    featMat[ , 6, ] <- array(img_pad[abind(rep(pad_row-1, 3*width), rep(rep(pad_col+1, rep(height,width)),3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    featMat[ , 7, ] <- array(img_pad[abind(rep(pad_row, 3*width), rep(rep(pad_col+1, rep(height,width)),3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    featMat[ , 8, ] <- array(img_pad[abind(rep(pad_row+1, 3*width), rep(rep(pad_col+1, rep(height,width)),3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    
    
    ### step 1. for each pixel and each channel in imgLR:
    ###           save (the neighbor 8 pixels - central pixel) in featMat
    ###           tips: padding zeros for boundary points
    
    ### step 2. apply the modelList over featMat
    predMat <- test(modelList, featMat)
    predMat <- predMat + abind(centre, centre, centre, centre, along = 2)
    
    ### step 3. recover high-resolution from predMat and save in HR_dir
    imgHR <- array(NA, c(height*2, width*2, 3))
    predMat_new <- array(aperm(predMat, c(2,1,3)), c(2,2,n_points,3))
    predMat_new <- aperm(array(aperm(predMat_new, c(2,1,3,4)), c(2,n_points*2,3)), c(2,1,3))
    imgHR[,seq(1,2*width,2),] = predMat_new[,1,]
    imgHR[,seq(2,2*width,2),] = predMat_new[,2,]
    
    imgHR = Image(imgHR, colormode = Color)
    writeImage(imgHR, paste0(HR_dir,  "img", "_", sprintf("%04d", i), ".jpg"))
  }
}