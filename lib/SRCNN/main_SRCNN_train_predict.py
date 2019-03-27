import os
import cv2
import h5py
import math
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
import time
#Build trainning model
def train_model(n1, n2,n3, k1 , k2 , k3,Patch_size, learn_rate,channel):
    SRCNN = tf.keras.Sequential()
    SRCNN.add(tf.keras.layers.Conv2D(n1, kernel_size=k1,activation='relu', padding='same',input_shape=(Patch_size, Patch_size, channel)))#patch size
    SRCNN.add(tf.keras.layers.Conv2D(n2, kernel_size=k2,activation='relu', padding='same'))
    SRCNN.add(tf.keras.layers.Conv2D(n3, kernel_size=k3,activation='linear', padding='same'))
    adam = tf.keras.optimizers.Adam(lr=learn_rate)
    SRCNN.compile(optimizer=adam, loss='mse')
    return SRCNN


#training
def training(train_data,train_label,model,batch,epoch):
    train_start_time = time.time()
    training = model.fit(train_data,train_label,batch_size=batch,epochs=epoch,validation_split=0.2)
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    return training, train_time


def show_psnr(model,valid):
    if valid == True:
        valid_mse = model.history['val_loss'][-1]
        valid_psnr = 20 * math.log10(1.) - 10 * math.log10(valid_mse)
        train_mse = model.history['loss'][-1]
        train_psnr = 20 * math.log10(1.) - 10 * math.log10(train_mse)
        print('Training mse: {}   psnr : {}\nValidation mse: {} psnr : {} '.format(train_mse,train_psnr,valid_mse,valid_psnr))
    else:
        train_mse = model.history['loss'][-1]
        train_psnr = 20 * math.log10(1.) - 10 * math.log10(train_mse)
        print('Training mse: {}   psnr : {}\n'.format(train_mse,train_psnr))

    
    
def show_error(model,valid):
    if valid == True:
        fig=plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.plot(model.history['loss'],label='Train loss')
        ax.plot(model.history['val_loss'],label = 'Validation loss')
        ax.legend(loc=0)
        plt.title('mean square error over 100 epoch')
        plt.xlabel('epoch')
        plt.ylabel('mse')
    
    else:
        fig=plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.plot(model.history['loss'],label='Train loss')
        ax.legend(loc=0)
        plt.title('mean square error over 100 epoch')
        plt.xlabel('epoch')
        plt.ylabel('mse')
        
    
#Save training model weights
def save_weight(model,path):
    model.save_weights(path)
    
    


#Build predict model
def predict_model(n1, n2,n3, k1 , k2 , k3, learn_rate,channel):
    SRCNN = tf.keras.Sequential()
    SRCNN.add(tf.keras.layers.Conv2D(n1, kernel_size=k1,activation='relu', padding='same',input_shape=(None, None, channel)))#patch size
    SRCNN.add(tf.keras.layers.Conv2D(n2, kernel_size=k2,activation='relu', padding='same'))
    SRCNN.add(tf.keras.layers.Conv2D(n3, kernel_size=k3,activation='linear', padding='same'))
    adam = tf.keras.optimizers.Adam(lr=learn_rate)
    SRCNN.compile(optimizer=adam, loss='mse')
    return SRCNN

    
    
    
def predicting(SRCNN_pred,n_test_files,test_lr_dir,test_hr_dir,test_lr_name,test_super_dir,scale):
    #predict SR
    predict_start_time= time.time()

    for i in range(n_test_files):
        test_lr_img_path= test_lr_dir + test_lr_name[i]
        test_lr_img = cv2.imread(test_lr_img_path,cv2.IMREAD_COLOR) 
        test_lr_img = cv2.cvtColor(test_lr_img,cv2.COLOR_BGR2YCrCb) #convert to YCrCb color type  
        t_shape = test_lr_img.shape

        #double the size of LR by cubic interpolation
        test_lr_img = cv2.resize(test_lr_img,(t_shape[1]*scale,t_shape[0]*scale),interpolation = cv2.INTER_CUBIC)
        new_shape = test_lr_img.shape

        #Construct feature dimension for CNN (1*height*width*1)
        Y_img = test_lr_img 
        Y = numpy.zeros((1, new_shape[0], new_shape[1], 3), dtype=float)
        Y[0, :, :, :] = Y_img.astype(float) / 255.     

        #predict super resolution images
        pre = SRCNN_pred.predict(Y, batch_size=1) * 255.  
        pre[pre[:] > 255] = 255
        pre[pre[:] < 0] = 0 
        pre = pre.astype(numpy.uint8)# integer (0 to 255) type
        pre = pre[0,:,:,:]
        pre = cv2.cvtColor(pre,cv2.COLOR_YCrCb2BGR) #convert color back to BGR type


        #Write SR images
        cv2.imwrite(os.path.join(test_super_dir, test_lr_name[i]), pre)

    predict_end_time = time.time()
    predict_time = predict_end_time - predict_start_time 

    return predict_time
    
    

#get test image PSNR 
def get_psnr(HR,SR):
    HR = numpy.array(HR,dtype = float)
    SR = numpy.array(SR,dtype = float)
    DIFF = SR - HR
    DIFF = DIFF.flatten('C')    
    MSE =numpy.mean(DIFF ** 2.)
    PSNR = 20 * math.log10(255.) - 10 * math.log10(MSE)
    return PSNR




    