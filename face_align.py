import numpy as np

from keras.layers import Conv2D,Input,Flatten,Dense,TimeDistributed,Reshape,Merge,BatchNormalization,Dropout
from keras.layers import MaxPooling2D
from keras.models import Model
import tensorflow as tf

def face_landmark(img_input):
    ## the input is Rhm_feature
    #img_input = Input(shape=(176, 176, 32))
    # Block 1
    x = Conv2D(48, (3, 3), activation='relu', padding='same', name='align_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='align_conv2')(x)
    #x = Conv2D(128, (3, 3), activation='relu', padding='same', name='align_conv4')(x)
    x = BatchNormalization()(x)
    x1= Conv2D(64, (3, 3), activation='relu', padding='same', name='align_conv3')(x)
    x1 = BatchNormalization()(x1)
    x2=Flatten()(x1)
    print('the shape of x2 in face_alin is ',x2.shape)
    x=Dense(256,activation='relu')(x2)
    x=Dense(132,name='landmark_pred')(x)


    #model=Model(inputs=img_input,outputs=x)
    # model.compile(optimizer='sgd',loss=euc_loss,metrics='mse')

    return x,x1



