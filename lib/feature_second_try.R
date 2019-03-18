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
    
    ### step 1. sample n_points from imgLR
    height <- dim(imgLR)[1]; width <- dim(imgLR)[2] # Height & width of imageLR
    ptidx <- sample(x = seq(height * width), size = n_points, replace = FALSE) # Index of sampling points 
    ptrow <- (ptidx - 1) %% height + 1 # Rows of sampling points
    ptcol <- (ptidx - 1) %/% height + 1 # Columns of sampling points

    ### step 2. for each sampled point in imgLR,
    
        ### step 2.1. save (the neighbor 8 pixels - central pixel) in featMat
        ###           tips: padding zeros for boundary points
    imgLR_mat <- matrix(aperm(imgLR, c(3,1,2)), nrow = 3) # Turn imageLR into matrix: channel * (height*width)
    imgLR_pad <- cbind(cbind(matrix(0L, nrow = 3, ncol = height+1),
                             imgLR_mat),
                       matrix(0L, nrow = 3, ncol = height+1)) # Add padding
    
    idxtop <- which(ptrow == 1)
    idxbtm <- which(ptrow == height)
    idxlft <- which(ptcol == 1)
    idxrgt <- which(ptcol == width)
    
    padidx <- ptidx + height + 1 # Index of sampling points in padding matrix
    ptngb <- array(c(imgLR_pad[, padidx - height - 1], imgLR_pad[, padidx - height], imgLR_pad[, padidx - height + 1],
                     imgLR_pad[, padidx - 1], imgLR_pad[, padidx + 1],
                     imgLR_pad[, padidx + height - 1], imgLR_pad[, padidx + height], imgLR_pad[, padidx + height + 1]),
                   c(3, n_points, 8)) # 8 neighbour pixels
    
    fm = aperm(ptngb, c(2, 3, 1))
    pt <- array(c(imgLR[ , , 1][ptidx], imgLR[ , , 2][ptidx], imgLR[ , , 3][ptidx]),c(n_points, 3)) # Central pixel
    fm <- sweep(fm, c(1,3), pt) # Minus central pixel
    
    featMat[((i-1)*n_points+1):(i*n_points), , ] = fm
    featMat[(i-1)*n_points + idxtop, c(1,4,6),] = 0
    featMat[(i-1)*n_points + idxbtm, c(3,5,8),] = 0
    featMat[(i-1)*n_points + idxlft, c(1,2,3),] = 0
    featMat[(i-1)*n_points + idxrgt, c(6,7,8),] = 0
    
        ### step 2.2. save the corresponding 4 sub-pixels of imgHR in labMat
    channel1 <- array(c(imgHR[ , , 1][cbind(ptrow*2-1, ptcol*2-1)], imgHR[ , , 1][cbind(ptrow*2, ptcol*2-1)], 
                        imgHR[ , , 1][cbind(ptrow*2-1, ptcol*2)], imgHR[ , , 1][cbind(ptrow*2, ptcol*2)]),
                      c(n_points, 4))
    channel2 <- array(c(imgHR[ , , 2][cbind(ptrow*2-1, ptcol*2-1)], imgHR[ , , 2][cbind(ptrow*2, ptcol*2-1)], 
                        imgHR[ , , 2][cbind(ptrow*2-1, ptcol*2)], imgHR[ , , 2][cbind(ptrow*2, ptcol*2)]),
                      c(n_points, 4))
    channel3 <- array(c(imgHR[ , , 3][cbind(ptrow*2-1, ptcol*2-1)], imgHR[ , , 3][cbind(ptrow*2, ptcol*2-1)], 
                        imgHR[ , , 3][cbind(ptrow*2-1, ptcol*2)], imgHR[ , , 3][cbind(ptrow*2, ptcol*2)]),
                      c(n_points, 4))
    labMat[((i-1)*n_points+1):(i*n_points), , ] = array(c(channel1, channel2, channel3), c(n_points, 4, 3))
    
    ### step 3. repeat above for three channels
    
  }
  return(list(feature = featMat, label = labMat))
}
