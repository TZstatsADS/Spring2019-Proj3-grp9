import os
import cv2
import h5py
import math
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
import time

def feature_RGB(n_files,Random_Crop,Patch_size,train_LR_dir,train_HR_dir,LR_dir_name,HR_dir_name):    
    feature_start_time = time.time()
    #Set feature and label dimension
    train_data = numpy.zeros((n_files * Random_Crop,  Patch_size, Patch_size,3), dtype=numpy.double) #feature size 33 * 33 * 3
    train_label = numpy.zeros((n_files * Random_Crop,  Patch_size, Patch_size,3), dtype=numpy.double) #label size 33 * 33 * 3

    for i in range(n_files):
    
        LR_file_path = train_LR_dir + LR_dir_name[i] #Low Resolution image path
        HR_file_path = train_HR_dir + HR_dir_name[i] #High Resolution image path
    
        hr_img = cv2.imread(HR_file_path,cv2.IMREAD_COLOR) #Read High Resolution image as label
        shape = hr_img.shape #High Resolution shape
    
        lr_img = cv2.imread(LR_file_path,cv2.IMREAD_COLOR)# Read low Resolution image as feature
        lr_img = cv2.resize(lr_img,(shape[1],shape[0]),interpolation = cv2.INTER_CUBIC) #Resize the LR to HR size using cubic interpolation

        #Select random points
        numpy.random.seed(2019)
        Points_x = numpy.random.randint(0, shape[0]- Patch_size, Random_Crop)
        Points_y = numpy.random.randint(0, shape[1]- Patch_size, Random_Crop)   

        #Construct patches by random points
        for j in range(Random_Crop):
            lr_patch = lr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size,:]
            hr_patch = hr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size,:]

        #Convert pixel range from (0,255) to (0,1)
            lr_patch = lr_patch.astype(float) / 255.
            hr_patch = hr_patch.astype(float) / 255.

        #Construct feature and label matrix
            train_data[i * Random_Crop + j,  :, :,:] = lr_patch
            train_label[i * Random_Crop + j, :, :,:] = hr_patch

    feature_end_time = time.time()
    feature_time = feature_end_time - feature_start_time
    return train_data,train_label,feature_time



def feature_YCrCb(n_files,Random_Crop,Patch_size,train_LR_dir,train_HR_dir,LR_dir_name,HR_dir_name):    
    feature_start_time = time.time()
    #Set feature and label dimension
    train_data = numpy.zeros((n_files * Random_Crop,  Patch_size, Patch_size,3), dtype=numpy.double) #feature size 32 * 32 * 3
    train_label = numpy.zeros((n_files * Random_Crop,  Patch_size, Patch_size,3), dtype=numpy.double) #label size 32 * 32 * 3

    for i in range(n_files):
    
        LR_file_path = train_LR_dir + LR_dir_name[i] #Low Resolution image path
        HR_file_path = train_HR_dir + HR_dir_name[i] #High Resolution image path
    
        hr_img = cv2.imread(HR_file_path,cv2.IMREAD_COLOR) #Read High Resolution image as label
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb) #Convert RGB to YCrCb Color type
        shape = hr_img.shape #High Resolution shape
    
        lr_img = cv2.imread(LR_file_path,cv2.IMREAD_COLOR)# Read low Resolution image as feature
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2YCrCb)#Convert RGB to YCrCb Color type
        lr_img = cv2.resize(lr_img,(shape[1],shape[0]),interpolation = cv2.INTER_CUBIC) #Resize the LR to HR size using cubic interpolation

        #Select random points
        numpy.random.seed(2019)
        Points_x = numpy.random.randint(0, shape[0]- Patch_size, Random_Crop)
        Points_y = numpy.random.randint(0, shape[1]- Patch_size, Random_Crop)   

        #Construct patches by random points
        for j in range(Random_Crop):
            lr_patch = lr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size,:]
            hr_patch = hr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size,:]

        #Convert pixel range from (0,255) to (0,1)
            lr_patch = lr_patch.astype(float) / 255.
            hr_patch = hr_patch.astype(float) / 255.

        #Construct feature and label matrix
            train_data[i * Random_Crop + j,  :, :,:] = lr_patch
            train_label[i * Random_Crop + j, :, :,:] = hr_patch
            
    feature_end_time = time.time()
    feature_time = feature_end_time - feature_start_time
    return train_data,train_label,feature_time



def feature_Y_color(n_files,Random_Crop,Patch_size,train_LR_dir,train_HR_dir,LR_dir_name,HR_dir_name):    
    feature_start_time = time.time()
    #Set feature and label dimension
    train_data = numpy.zeros((n_files * Random_Crop,  Patch_size, Patch_size,1), dtype=numpy.double) #feature size 32 * 32 * 1
    train_label = numpy.zeros((n_files * Random_Crop,  Patch_size, Patch_size,1), dtype=numpy.double) #label size 32 * 32 * 1

    for i in range(n_files):
    
        LR_file_path = train_LR_dir + LR_dir_name[i] #Low Resolution image path
        HR_file_path = train_HR_dir + HR_dir_name[i] #High Resolution image path
    
        hr_img = cv2.imread(HR_file_path,cv2.IMREAD_COLOR) #Read High Resolution image as label
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb) #Convert RGB to YCrCb Color type
        hr_img = hr_img[:, :, 0] #Use Y color channel for trainning
        shape = hr_img.shape #High Resolution shape
    
        lr_img = cv2.imread(LR_file_path,cv2.IMREAD_COLOR)# Read low Resolution image as feature
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2YCrCb)#Convert RGB to YCrCb Color type
        lr_img = lr_img[:, :, 0]#Use Y color channel for trainning
        lr_img = cv2.resize(lr_img,(shape[1],shape[0]),interpolation = cv2.INTER_CUBIC) #Resize the LR to HR size using cubic interpolation

        #Select random points
        numpy.random.seed(2019)
        Points_x = numpy.random.randint(0, shape[0]- Patch_size, Random_Crop)
        Points_y = numpy.random.randint(0, shape[1]- Patch_size, Random_Crop)   

        #Construct patches by random points
        for j in range(Random_Crop):
            lr_patch = lr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]
            hr_patch = hr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]

        #Convert pixel range from (0,255) to (0,1)
            lr_patch = lr_patch.astype(float) / 255.
            hr_patch = hr_patch.astype(float) / 255.

        #Construct feature and label matrix
            train_data[i * Random_Crop + j,  :, :,0] = lr_patch
            train_label[i * Random_Crop + j, :, :,0] = hr_patch
            
    feature_end_time = time.time()
    feature_time = feature_end_time - feature_start_time
    return train_data,train_label,feature_time



#write h5 file for feature and label construction
def write_h5py(train_feature,train_label,path):
    x = train_feature.astype(numpy.float32)
    y = train_label.astype(numpy.float32)
    with h5py.File(path, 'w') as h:
        h.create_dataset('train_feature', data=x, shape=x.shape)
        h.create_dataset('train_label', data=y, shape=y.shape)
        
        
        
# reading saved h5py train feature file
def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        feature = numpy.array(hf.get('train_feature'))
        label = numpy.array(hf.get('train_label'))
        return feature, label
    


        

