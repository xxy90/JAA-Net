import numpy as np

from keras.layers import Conv2D,Input,Flatten,Dense,TimeDistributed,Reshape,Merge,BatchNormalization,Dropout,Lambda
from keras.layers import MaxPooling2D,merge,multiply,concatenate
from keras.models import Model,Sequential
import tensorflow as tf
import numpy as np
import cv2

def multi_scale_learning(img_input):
    ## input is l*l*3,achieve AU and landmark detecttion simutaniously

    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x=BatchNormalization()(x)    
    x = Conv2D(48, (3, 3), activation='relu', padding='same', name='block1_conv4')(x)
    #x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv3')(x)
    xx = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    xx = BatchNormalization()(xx)
    xx = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(xx)

    print('the shape of  xx is ',xx.shape)


    # the patch of 8*8
    patchs1= Lambda(lambda x:get_patchs(x,8))(xx)
    print('the shape of patchs1 is ',patchs1.shape)   
    scales1=TimeDistributed(Conv2D(16,(3,3),padding='same'))(patchs1)## the shape is (num-data,num_patch1,L/8,L/8,16)
        
    ful_scales1 = Reshape((88, 88, 16))(scales1)

    ## second patch is 4*4
    patchs2 =  Lambda(lambda x:get_patchs(x, 4))(ful_scales1)
    scales2 = TimeDistributed(Conv2D(8, (3, 3), padding='same'))(patchs2)## the shape is (num-data,num_patch2,L/4,L/4,8)
   
    ful_scales2 = Reshape((88, 88, 8))(scales2)

    ## third patch is 2*2
    patchs3 = Lambda(lambda x:get_patchs(x, 2))(ful_scales2)
    scales3 = TimeDistributed(Conv2D(8, (3, 3), padding='same'))(patchs3)## the shape is (num-data,num_patch3,L/2,L/2,8)
    
    ful_scales3 = Reshape(( 88, 88, 8))(scales3)
    

   #Rhm_feature=merge((ful_scales1,ful_scales2,ful_scales3),mode='concat') ## concate in the channel direction
    Rhm_feature=concatenate([ful_scales1,ful_scales2,ful_scales3,xx],axis=3)
    Rhm_feature = MaxPooling2D((2, 2), strides=(2, 2), name='rhm_pool')(Rhm_feature)
    #Rhm_feature =padding_removal(Rhm_feature,6)


    return Rhm_feature

def padding_removal(data,scale):

    L = data.get_shape().as_list()[1]
    ## upscaling the data
    data=cv2.resize(data,(L+scale,L+scale),interpolation=cv2.INTER_LINEAR)
    ## remove the bounding to original shape
    data=data[3:L+3,3:L+3,:]


    return data


def get_patchs(data,size):
    ## the data shape is 176*176,size is 8,4,2

    L=data.get_shape().as_list()[1]
    D=data.get_shape().as_list()[3]


    patchs=tf.zeros((int(L/size),int(L/size),D))
    
    patchs=[]
    #print("the patchs shape is", patchs.shape)
    for i in range(0,int(L/size)):
        for j in range(0,int(L/size)):
            tmp_patch=data[:,i*size:(i+1)*size,j*size:(j+1)*size,:]
            
            patchs.append(tmp_patch)


    
    patchs=tf.convert_to_tensor(patchs)
    patchs=tf.transpose(patchs,(1,0,2,3,4))
    #print('the shape of patchs is ',patchs.shape)
    


    return patchs

def merge_patchs(data, size):

    L = data.get_shape().as_list()[1]
    D = data.get_shape().as_list()[3]

    L = int(np.sqrt(int(L)))
    #patchs = tf.zeros((int(L * size), int(L * size), D))
    patchs = np.zeros((int(L * size), int(L * size), D))
    patch_list = []
    for c in range(0, L * L):
        patch = data[:, c, :, :, :]
        patch_list.append(patch)


    # for i in range(0, L):
    #     for j in range(0, L ):
    #
    #             patchs[i * size:(i + 1) * size, j * size:(j + 1) * size, :] = patch_list[i]
    #             #patchs[:,i * size:(i+1)*size,j*size:(j+1)*size,:] = data[:, c,:,:,:]


    patchs = tf.convert_to_tensor(patch_list)
    #patchs = tf.transpose(patchs, (1, 0, 2, 3, ))
    print('the shape of patchs is ',patchs.shape)

    return patchs




