#############################################################
### Construct features and responses for training images###
#############################################################

### Author: Chao Yin
### Project 3

feature <- function(LR_dir, HR_dir, n_points=1000){
  
  ### Construct process features for training images (LR/HR pairs)
  
  ### Input: a path for low-resolution images + a path for high-resolution images 
  ###        + number of points sampled from each LR image
  ### Output: an .RData file contains processed features and responses for the images
  
  ### load libraries
  library("EBImage")
  
  n_files <- length(list.files(LR_dir))
  
  ### store feature and responses
  featMat <- array(NA, c(n_files * n_points, 8, 3))
  labMat <- array(NA, c(n_files * n_points, 4, 3))
  
  ### read LR/HR image pairs
  for(i in 1:n_files){
    imgLR <- readImage(paste0(LR_dir,  "img_", sprintf("%04d", i), ".jpg"))
    imgHR <- readImage(paste0(HR_dir,  "img_", sprintf("%04d", i), ".jpg"))
    
    height <- dim(imgLR)[1] 
    width <- dim(imgLR)[2]
    ptrow <- sample(x = height, size = n_points, replace = TRUE)
    ptcol <- sample(x = width, size = n_points, replace = TRUE)
    centre <- array(imgLR[abind(rep(ptrow, 3), rep(ptcol, 3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points , 1, 3))
    img_pad <- abind(array(0, c(height, 1, 3)), imgLR, array(0, c(height, 1, 3)), along = 2)
    img_pad <- abind(array(0, c(1, width + 2, 3)), img_pad, array(0, c(1, width + 2, 3)), along = 1)
    
    low_i <- (i - 1)*n_points + 1
    high_i <- i*n_points
    pad_row <- ptrow + 1
    pad_col <- ptcol + 1
    
    featMat[low_i:high_i, 1, ] <- array(img_pad[abind(rep(pad_row-1, 3), rep(pad_col-1, 3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    featMat[low_i:high_i, 2, ] <- array(img_pad[abind(rep(pad_row, 3), rep(pad_col-1, 3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    featMat[low_i:high_i, 3, ] <- array(img_pad[abind(rep(pad_row+1, 3), rep(pad_col-1, 3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    featMat[low_i:high_i, 4, ] <- array(img_pad[abind(rep(pad_row-1, 3), rep(pad_col, 3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    featMat[low_i:high_i, 5, ] <- array(img_pad[abind(rep(pad_row+1, 3), rep(pad_col, 3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    featMat[low_i:high_i, 6, ] <- array(img_pad[abind(rep(pad_row-1, 3), rep(pad_col+1, 3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    featMat[low_i:high_i, 7, ] <- array(img_pad[abind(rep(pad_row, 3), rep(pad_col+1, 3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    featMat[low_i:high_i, 8, ] <- array(img_pad[abind(rep(pad_row+1, 3), rep(pad_col+1, 3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    
    labMat[low_i:high_i, 1, ] <- array(imgHR[abind(rep(ptrow*2-1, 3), rep(ptcol*2-1, 3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    labMat[low_i:high_i, 2, ] <- array(imgHR[abind(rep(ptrow*2, 3), rep(ptcol*2-1, 3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    labMat[low_i:high_i, 3, ] <- array(imgHR[abind(rep(ptrow*2-1, 3), rep(ptcol*2, 3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
    labMat[low_i:high_i, 4, ] <- array(imgHR[abind(rep(ptrow*2, 3), rep(ptcol*2, 3), rep(1:3, rep(n_points, 3)), along = 2)], c(n_points, 1, 3)) - centre
  }
  return(list(feature = featMat, label = labMat))
}
