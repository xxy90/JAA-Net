from AU_detect import AU_detect,AU_detect2,AU_detect3
from input_data import get_data_labels_landmark,get_disfa_data_labels_landmark,get_transformed_data_labels_landmark,get_ck_data_labels_landmark
import os 
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='4'
#os.environ['CUDA_VISIBLE_DEVICES']='4'
#config.gpu_options.per_process_gpu_memory_fraction = 0.5 

#config.gpu_options.allow_growth = True 

## use disfa data to train model

def train():
    train_data,train_label,landmark,names=get_data_labels_landmark()
    #train_data,train_label,landmark,loss_true=get_disfa_data_labels_landmark()
    

    print('the type of datas are',type(train_data),type(train_label),type(landmark) )
    model=AU_detect()

    model.fit(train_data,[landmark,train_label],verbose=1,batch_size=9,epochs=50,shuffle=True)

    model.save_weights('JAA_NET_ck.npy')

    return model
def train_with_attention_loss():
    #train_data,train_label,landmark,loss_true=get_data_labels_landmark()

    print('the type of datas are',type(train_data),type(train_label),type(landmark) )
    model=AU_detect2()

    model.fit(train_data,[landmark,train_label,loss_true],verbose=1,batch_size=4,epochs=10,shuffle=True)

    model.save_weights('JAA_NET_attention.npy')

    return model
def val():
    train_data, train_label, landmark, loss_true = get_data_labels_landmark()
    np.random.seed(5)
    np.random.shuffle(train_data)
    np.random.seed(5)
    np.random.shuffle(train_label)
    np.random.seed(5)
    np.random.shuffle(landmark)
    num = int(np.ceil(train_data.shape[0] * 0.8))
    train = train_data[:num]
    traininglabel = train_label[:num]
    trainlandmark=landmark[:num]
    val = train_data[num:]
    vallabel = train_label[num:]
    vallandmark=landmark[num:]

    model = AU_detect()

    model.fit(train, [trainlandmark, traininglabel], 
validation_data=(val,[vallandmark,vallabel]),verbose=1,epochs=100,batch_size=9)
    model.save_weights('JAA_NET_val.npy')
    model.save('JAA_NET_val.h5')

    return model
   
if __name__=='__main__':
    #train()
    val()
    #train_with_attention_loss()