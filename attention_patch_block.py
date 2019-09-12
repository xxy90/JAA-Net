from keras.layers import merge,Conv2D,MaxPooling2D,Dense,BatchNormalization
from keras.layers.core import *

from keras.models import *
import tensorflow as tf


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    TIME_STEPS = 16  ##
    a = Lambda(lambda x: tf.transpose(x, (0, 2, 3, 4, 1)))(inputs)  ##(batch_size,11,11,48,time_step)  

    
    a = Flatten()(a)
    a = Dense(TIME_STEPS, activation='softmax')(a)  ## the shape a [batch_size,time_step]
    a = Lambda(lambda x: tf.expand_dims(x, 1))(a)
    a = Lambda(lambda x: tf.expand_dims(x, 1))(a)
    a = Lambda(lambda x: tf.expand_dims(x, 1))(a)

    a = Lambda(lambda x: tf.tile(x, (1, 11, 11, 48, 1)))(a)

    a = Lambda(lambda x: tf.transpose(x, (0, 4, 1, 2, 3)))(a)
    # output_attenion_mul=Lambda(lambda x,y:tf.multiply(x,y))(inputs,a)
    output = merge([inputs, a], name='attention_mul', mode='mul')
    return output

def attention_whole_block(inputs):
    # inputs.shape = (batch_size, 176,176,3) get_weight for input_dim
    #TIME_STEPS = 16  ##
    

    a = Conv2D(256, (3, 3), padding='same')(inputs)  ## the shape a [batch_size,time_step]
    a= Conv2D(64, (3, 3), activation='relu', padding='same', name='attention_conv1')(a)
    a = BatchNormalization()(a)
    aa = MaxPooling2D((2, 2), strides=(2, 2), name='attention_pool')(a)
    aa = MaxPooling2D((2, 2), strides=(2, 2), name='attention_pool2')(aa)
    
    return aa,a

def attention_whole_block2(inputs):
    # inputs.shape = (batch_size, 176,176,3) get_weight for input_dim

    a = Conv2D(48, (3, 3), padding='same', name='after_weight_whole')(inputs)  ## the shape a [batch_size,time_step]

    return a
