from keras.layers import Conv2D,Input,Flatten,Dense,TimeDistributed,Reshape,Merge,Dropout,Lambda,UpSampling2D
from keras.layers import MaxPooling2D,merge,concatenate,multiply,subtract,division,BatchNormalization
from keras.models import Model
import tensorflow as tf
from face_align import face_landmark
import numpy as np
from pre_learing import multi_scale_learning
import keras.backend as K
from attention_patch_block import attention_3d_block,attention_whole_block,attention_whole_block2
from keras.utils import multi_gpu_model
from keras import optimizers
from utils import get_au_point

def global_feature(img_input):
    ## input is Rhm-feature
    #img_input = Input(shape=(176, 176, 32))
    print('the input of img_input is ',img_input.get_shape().as_list())
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='global_conv1')(img_input)
    #x = BatchNormalization()(x)
    x = Conv2D(48, (3, 3), activation='relu', padding='same', name='global_conv2')(x)
    #x = Conv2D(128, (3, 3), activation='relu', padding='same', name='global_conv4')(x)   
    
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='global_conv3')(x)
    x = BatchNormalization()(x)

    return x

def get_roi_patch(point,data):
    ## data is tensors[num_data,88,88,48]
    #point,x=face_landmark(data)

    d= Lambda(lambda x:tf.sqrt(tf.square(x[:,42*2] - x[:,40*2])+tf.square(x[:,42*2+1] - x[:,40*2+1])))(point)

    lis = [21, 22, 18, 25, 19, 24, 41, 46, 36, 44, 50, 48, 54, 56, 58, 51, 57]
    pp=[]
    ## AU1
    pp.append(point[:,21*2])
    pp.append(point[:,21*2+1]+d/2)

    pp.append(point[:, 22 * 2])
    pp.append(point[:, 22 * 2 + 1] + d / 2)

    pp.append(point[:, 18 * 2])
    pp.append(point[:, 18 * 2 + 1] + d / 3)


    pp.append(point[:, 25 * 2])
    pp.append(point[:, 25 * 2 + 1] + d / 3)

    pp.append(point[:, 19 * 2])
    pp.append(point[:, 19 * 2 + 1] - d / 2)

    pp.append(point[:, 24 * 2])
    pp.append(point[:, 24 * 2 + 1] - d / 2)

    pp.append(point[:, 41 * 2])
    pp.append(point[:, 41 * 2 + 1] - d / 2)

    pp.append(point[:, 46 * 2])
    pp.append(point[:, 46 * 2 + 1] - d / 2)
    # AU7
    pp.append((point[:,36*2]+point[:,39*2])/2)
    pp.append((point[:,36*2 +1]+point[:,39*2+ 1])/2)
    pp.append((point[:,42*2]+point[:,44*2 ])/2)
    pp.append((point[:,42*2+1]+point[:,44*2+ 1])/2)

    pp.append(point[:, 50 * 2])
    pp.append(point[:, 50 * 2 + 1] )

    pp.append(point[:, 48 * 2])
    pp.append(point[:, 48 * 2 + 1] )

    pp.append(point[:, 54 * 2])
    pp.append(point[:, 54 * 2 + 1])

    pp.append(point[:, 56 * 2])
    pp.append(point[:, 56 * 2 + 1] - d / 2)

    pp.append(point[:, 58 * 2])
    pp.append(point[:, 58 * 2 + 1] - d / 2)

    pp.append(point[:, 41 * 2])
    pp.append(point[:, 41 * 2 + 1] )

    # pp.append(point[:, 46 * 2])
    # pp.append(point[:, 46 * 2 + 1] )


    pp=Lambda(lambda x:tf.convert_to_tensor(x))(pp)## shape is [17*2,num_data]
    pp=Lambda(lambda x:tf.transpose(x,(1,0)))(pp)## shape is [num_data,17*2]
    #conv_fea=multi_scale_learning(data)## is [num_data,88,88,48]
    point_for_conv = Lambda(lambda x:x/2)(pp)
    point_for_conv=Lambda(lambda x:x-5)(point_for_conv) ## the start of image slice

    point_for_conv=Lambda(lambda x:tf.expand_dims(x,axis=2))(point_for_conv)
    point_for_conv =Lambda(lambda x: tf.expand_dims(x, axis=3))(point_for_conv)


    point_for_conv=Lambda(lambda x:tf.tile(x,(1,1,32,48)))(point_for_conv)
    point_for_conv=Lambda(lambda x:tf.cast(x,dtype=tf.int32))(point_for_conv)

    f=[]
    for j in range(16):
        tmp_conv_fea=data
        # x=tf.to_int64(point_for_conv[:,2*j])
        # y=tf.to_int64(point_for_conv[:,2*j+1])
        # tmp=tmp_conv_fea[:,x-5:x+6,y-5:y+6,:]

        tmp=tf.slice(tmp_conv_fea,point_for_conv[:,j*2,j*2+1,0],[-1,11,11,48])
        f.append(tmp)
    f=Lambda(lambda x:tf.convert_to_tensor(x))(f)
    f=Lambda(lambda x:tf.transpose(x,(1,0,2,3,4)))(f)
    print('the shape of f is',f.shape)

    return f,pp

def get_w2(point):
    ## the shape of w is [num_data,16,176,176,3]
    pp=get_au_point(point)
    print('the shape of pp is ', pp.shape)
    w=[]
    for k in range(16):
        init_w = []
        for i in range(176):
            for j in range(176):
                tmp = 1 - (tf.abs(pp[:, 2*k] - i) + tf.abs(pp[:, 2*k+1] - j))/ 176
                zeros = tf.zeros_like(tmp)
                init_w.append(tf.reduce_max((tmp, zeros), axis=0))

        init_w=tf.convert_to_tensor(init_w)
        init_w=tf.transpose(init_w,(1,0))
        init_w=tf.reshape(init_w,(-1,176,176))
        w.append(init_w)

    w = tf.convert_to_tensor(w)
    print('the shape of w is ', w.shape)
    w = tf.transpose(w, (1, 0, 2, 3))

    w = tf.expand_dims(w, axis=4)
    w = tf.tile(w, (1, 1, 1, 1,3))

    return w






def get_w3(pp):
    ## the org_data is [num_data,176,176,3],pp is [num_data,32],for the whole patch w is [num_data,176,176,3]
    pp=tf.reshape(pp,(-1,66,2))

    #print('the shape of pp is ', pp.shape)
    w=[]
    pp=pp/4
    for i in range(44):
        for j in range(44):
            tmp=1-tf.reduce_max((tf.abs(pp[:,:,0]-i)+tf.abs(pp[:,:,1]-j)),axis=1)/176
            zeros=tf.zeros_like(tmp)
            w.append(tf.reduce_max((tmp,zeros),axis=0))

    w = tf.convert_to_tensor(w)
    print('the shape of w is ',w.shape)
    w = tf.transpose(w, (1, 0))

    w = tf.reshape(w, (-1, 44, 44))

    w = tf.expand_dims(w, axis=3)
    w = tf.tile(w, (1, 1, 1, 48))

    return w


def AU_detect():
    ## input is the Rhm-feature
    img_input=Input(shape=(176,176,3))
    if not hasattr(img_input,'_keras_history'):
        print('this is not keras tensor')
    pre_fea=multi_scale_learning(img_input)
    global_fea=global_feature(pre_fea)
    align_pred,align_fea=face_landmark(pre_fea)      
    #align_pred=Lambda(lambda x:trans(x))(align_pred)


    weighted_fea,weighted_map=attention_whole_block(img_input) 

    au_fea=merge([pre_fea,weighted_fea], name='attention_mul', mode='mul')
    
    ful_fea=merge([au_fea, global_fea, align_fea],mode='concat')
    ful_fea= Conv2D(256, (3, 3), activation='relu', padding='same', name='local_conv1')(ful_fea)
    ful_fea= Conv2D(128, (3, 3), activation='relu', padding='same', name='local_conv2')(ful_fea)
    ful_fea=Flatten()(ful_fea)    

    output=Dense(256,activation='relu')(ful_fea)

    pred=Dense(12,activation='tanh',name='au_pred')(output)

    model=Model(inputs=img_input,outputs=[align_pred,pred])
    #model=multi_gpu_model(model,4) 
    

    sgd = optimizers.SGD(lr=0.001, decay=5e-04, momentum=0.9)
    model.compile(optimizer='sgd',loss={'landmark_pred':euc_loss,'au_pred':au_loss},
                  loss_weights={'landmark_pred':1,'au_pred':0.5},
                  metrics={'landmark_pred':'mae','au_pred':f1})

    return model

def AU_detect2():
    ## input is the Rhm-feature assign weight to whole map
    img_input=Input(shape=(176,176,3))
    if not hasattr(img_input,'_keras_history'):
        print('this is not keras tensor')
    pre_fea=multi_scale_learning(img_input)
    global_fea=global_feature(pre_fea)
    align_pred,align_fea=face_landmark(pre_fea)
        
    init_w=Lambda(lambda x:get_w3(x))(align_pred)
    print('start multiply the init_w......')
    init_map=multiply([init_w,pre_fea])
    print('start process the attention block......')
    weighted_fea,weighted_map=attention_whole_block(init_map)
    attention_loss=Lambda(lambda x:(tf.reduce_mean(tf.square(x[0]-x[1]))),name='attention_loss')([weighted_map,init_map]) 
    #weighted_fea=Lambda(lambda x:tf.tile(x,(1,1,1,16)))(weighted_fea)
    au_fea=merge([pre_fea,weighted_map], name='attention_mul', mode='mul')
    
    ful_fea=merge([au_fea, global_fea, align_fea],mode='concat')

    ful_fea=Flatten()(ful_fea)    

    output=Dense(128,activation='relu')(ful_fea)

    pred=Dense(14,activation='tanh',name='au_pred')(output)
    model=Model(inputs=img_input,outputs=[align_pred,pred,attention_loss])
    #model=multi_gpu_model(model,gpus=4)  
    

    sgd = optimizers.SGD(lr=0.1, decay=1e-05, momentum=0.9)
    model.compile(optimizer='sgd',loss={'landmark_pred':euc_loss,'au_pred':au_loss,
'attention_loss':lambda y_true,attention_loss:attention_loss},
                  loss_weights={'landmark_pred':0.5,'au_pred':1,'attention_loss':1e-07},
                  metrics={'landmark_pred':'mse','au_pred':f1,'attention_loss':'mse'})

    return model

def AU_detect3():
    ## input is the Rhm-feature assign weight to whole map
    img_input = Input(shape=(176, 176, 3))
    if not hasattr(img_input, '_keras_history'):
        print('this is not keras tensor')
    pre_fea = multi_scale_learning(img_input)
    global_fea = global_feature(pre_fea)
    align_pred, align_fea = face_landmark(pre_fea)
    
    
    init_w = Lambda(lambda x: get_w2(x))(align_pred)
    org_input=Lambda(lambda x:tf.expand_dims(x,axis=1))(img_input)
    org_input=Lambda(lambda x:tf.tile(x,(1,16,1,1,1)))(org_input)
    init_map = multiply([init_w, org_input])
    weighted_fea,weighted_map = attention_whole_block2(init_map)
    attention_loss = Lambda(lambda x: (tf.reduce_mean(tf.square(x[0] - x[1]))), name='attention_loss')([init_map,weighted_map])

    weighted_fea=Lambda(lambda x:tf.transpose(x,(0,2,3,4,1)))(weighted_fea)
    weighted_fea=Reshape((44,44,48))(weighted_fea)
    au_fea = merge([pre_fea, weighted_fea], name='attention_mul', mode='mul')

    ful_fea = merge([au_fea, global_fea, align_fea], mode='concat')

    ful_fea = Flatten()(ful_fea)

    output = Dense(128, activation='relu')(ful_fea)

    pred = Dense(14, activation='tanh', name='au_pred')(output)

    model = Model(inputs=img_input, outputs=[align_pred, pred, attention_loss])
    #model=multi_gpu_model(model,4)


    sgd = optimizers.SGD(lr=1, decay=1e-04, momentum=0.9)#
    model.compile(optimizer='sgd',
                  loss={'landmark_pred': 'mean_squared_error', 'au_pred': au_loss,
 'attention_loss': lambda y_true,attention_loss:attention_loss},
                  loss_weights={'landmark_pred': 0.5, 'au_pred': 1, 'attention_loss': 1e-07},
                  metrics={'landmark_pred': 'mse', 'au_pred': f1})

    return model





def euc_loss(gt,pred):
    d=tf.sqrt(tf.reduce_sum(tf.square(gt[:,36*2:37*2]-gt[:,45*2:46*2])))

    rmse0 = tf.sqrt(tf.reduce_mean(tf.square(gt - pred)/d))

    return rmse0
def weight_loss(init_weight,model):

    after_weight = model.get_layer('attention_mul').get_weights()
    dif=tf.reduce_sum(K.categorical_crossentropy(init_weight,after_weight))
    return dif

def au_loss(label,pred):
    ## the shape of pred and label is (num_data,12)
    loss=0
    #R=get_rate_for_label(label)
    pred = tf.div(pred + 1, 2)
    label=(label+1)/2
    label = tf.convert_to_tensor(label)
    R = tf.reduce_sum(label, axis=0) / tf.reduce_sum(label)
    for i in range(12):
        temp_loss=-(label[:,i]*tf.log((pred[:,i]+0.05)/1.05)+(1-label[:,i])*tf.log((1.05-pred[:,i])/1.05))
        temp_sum=tf.reduce_mean(temp_loss)
        loss+=temp_sum
    f1 = micro_f1(label, pred)
    loss2 = tf.reduce_mean(tf.multiply(R, 1 - f1))
    #loss = loss + loss2
    loss=loss/12
    return loss

def micro_f1(label,pred):
    ##pred and label need to be tensor,pred is (-1,1)

    ## get the number of data_size and labels
    num_labels=12

    pos=tf.ones_like(pred)
    negs=tf.zeros_like(pred)-1
    zeros=tf.zeros_like(pred)
    pred_c=tf.where(tf.less(pred,0),negs,pos)

    if type(label)=='np.ndarray':
        label=tf.convert_to_tensor(label)
    ## compute precision
    TP_L = tf.where(tf.equal(label, 1), pos, negs)
    TP_P=tf.where(tf.equal(pred_c,1),pos,zeros)
    TP=tf.where(tf.equal(TP_L,TP_P),pos,zeros)

    FP_L=tf.where(tf.equal(label,-1),pos,negs)
    FP_P=tf.where(tf.equal(pred_c,1),pos,zeros)
    FP=tf.where(tf.equal(FP_L,FP_P),pos,zeros)

    FN_L = tf.where(tf.equal(label, 1), pos, negs)
    FN_P = tf.where(tf.equal(pred_c, -1), pos, zeros)
    FN = tf.where(tf.equal(FN_L, FN_P), pos, zeros)

    sum_TP_each_label=tf.reduce_sum(TP,axis=0)
    sum_FP_each_label=tf.reduce_sum(FP,axis=0)
    sum_FN_each_label=tf.reduce_sum(FN,axis=0)

    dem = tf.ones_like(sum_TP_each_label) * 1e-08
    sum_TP_each_label=tf.where(tf.equal(sum_TP_each_label,0),dem,sum_TP_each_label)

    sum_P=tf.add(sum_FP_each_label,sum_TP_each_label)+1e-08## tp+fp
    sum_R=tf.add(sum_TP_each_label,sum_FN_each_label)+1e-08## tp+fn

    precision=tf.div(sum_TP_each_label,sum_P)##(1,num_labels)

    recall = tf.div(sum_TP_each_label, sum_R)  ##(1,num_labels)
    avg_p=tf.div(tf.reduce_sum(precision),num_labels)
    avg_r=tf.div(tf.reduce_sum(recall),num_labels)

    micro_fl=tf.div(2*tf.multiply(avg_p,avg_r),tf.add(avg_p,avg_r))

    macro_f1=tf.div(2*tf.multiply(precision,recall),tf.add(precision,recall)+1e-08)##(1,num_labels)get f1 of each label


    return macro_f1

def get_rate_for_label(label):
    label=(label+1)/2
    n=label.shape[1]
    num=np.zeros((n,1))
    R=np.zeros(n,1)
    for i in range(n):
        num[i]=np.sum(label[:,i])
    R=num/np.sum(num)
    return R

def f1(y_true, y_pred):
    ## need the y_true{0,1}
    y_true=(y_true+1)/2
    y_pred=(y_pred+1)/2
    def recall(y_true, y_pred):
        """Recall metric.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
       
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

















