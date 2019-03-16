#############################################################
### Construct features and responses for training images###
#############################################################

### Authors: Chengliang Tang/Tian Zheng
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
    
    #sampling rows and columns
    sample_ind <- arrayInd(sample(length(imgLR@.Data[,,1]),n_points),dim(imgLR@.Data[,,1]))
    
        ### step 2. for each sampled point in imgLR,
        ### step 2.1. save (the neighbor 8 pixels - central pixel) in featMat
        ###           tips: padding zeros for boundary points
    
    ##padding 0 
    new_row = nrow(imgLR)+2
    new_col = ncol(imgLR) +2
    mat_pad = array(0,c(new_row,new_col))
    sample_ind_pad = sample_ind+1
    
    
    ## constuct rows and columns of 8 neighbors of X matrix
    up_left<-up<-up_right<-left<-right<-bottom_left<-bottom<-bottom_right<-sample_ind_pad
    up_left = up_left-1
    up[,1] = up[,1]-1
    up_right[,1] = up_right[,1]-1
    up_right[,2] = up_right[,2]+1    
    left[,2] = left[,2]-1
    right[,2] = right[,2]+1
    bottom_left[,1] = bottom_left[,1]+1
    bottom_left[,2] = bottom_left[,2]-1
    bottom[,1]=bottom[,1]+1
    bottom_right = bottom_right+1
    
    ## construct rows and columns of 4 pixels of y matrix
    H_low_right<-sample_ind
    H_low_right = H_low_right *2
    H_low_left<-H_up_left<-H_up_right<-H_low_right
    H_low_left[,2] = H_low_left[,2]-1
    H_up_left[,1] = H_up_left[,1]-1
    H_up_left[,2] = H_up_left[,2]-1
    H_up_right[,1] = H_up_right[,1]-1
              
    
    #Adding values to featMat and labMat in each channel
    for (j in 1:3) {
    
    #imgLR matrix for each channel after 0 padding
    mat_pad[2:(nrow(mat_pad)-1),2:(ncol(mat_pad)-1)]= imgLR@.Data[,,j]
    
    #imgHR matrix for each channel
    H_mat = imgHR@.Data[,,j]
    
    
    #Combining imgLR 8 neighbors to a single matrix
    my_neighbors = cbind(mat_pad[up_left],mat_pad[up],mat_pad[up_right],mat_pad[left],
                         mat_pad[right],mat_pad[bottom_left],mat_pad[bottom],mat_pad[bottom_right])
    
    #Combining imgHR 4 pixels  to a single matrix
    my_H_four = cbind(H_mat[H_up_left],H_mat[H_up_right],H_mat[H_low_left],H_mat[H_low_right])
    
    #Subtracting center pixels
    X_mat = my_neighbors - mat_pad[sample_ind_pad]
    y_mat = my_H_four - mat_pad[sample_ind_pad]
    
    
    #Update featMat and labMat
    if (i==1) {
      featMat[,,j][1:(i*n_points),] = X_mat
      labMat[,,j][1:(i*n_points),] = y_mat
    }
    else {
      featMat[,,j][(n_points*(i-1)+1):(n_points*i),] = X_mat
      labMat[,,j][(n_points*(i-1)+1):(n_points*i),] = y_mat
      
    }
  }
  }   

  return(list(feature = featMat, label = labMat))
}

