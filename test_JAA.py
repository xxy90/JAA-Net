import unittest


import time
from AU_detect import AU_detect
from input_data import get_data_labels_landmark
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf



def test():

    test_data,test_label,landmarks,names=get_data_labels_landmark()
    test_data=test_data[:5]
    test_label=test_label[:5]
    test_landmark=landmarks[:5]
    data=test_data[0]
    print('the test label is',test_label[0])
    data_n = data[np.newaxis,:]
    print('the shape of data_n is',data_n.shape)



    model=AU_detect()
    model.load_weights('JAA_NET_ck.npy')
    #score=model.evaluate(test_data,[test_landmark,test_label])
    landmark,label_pred=model.predict(data_n)
    show(data,landmark)

    label_paindata = [0, 4, 6, 7, 9, 10, 12, 15, 20, 25, 26, 27, 43, 50]  ##total 14 labels
    label_disfa = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]## total 12 labels
    label_bp4d =  [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]## total 12 labels
    En_name_list1 = ['Inner brow raiser', 'Outer brow raiser', 'Brow lowerer', 'Upper lid raiser',
                    'Cheek raiser', 'Nose wrinkler', 'Lip corner puller', 'Lip corner depressor', 'Chin raiser',                                                                                                      'Lip part',
                    'Lip stretcher', 'Lips part', 'Jaw drop']
    Ch_name_list1 = ['内眉提升', '外眉提升', '眉毛降低','上睑提升', '脸颊提升',
                    '皱鼻', '嘴角向上', '嘴角向下', '下巴提升',
                    '嘴角拉伸', '嘴唇微张','颌部下降']
    En_name_list2=['Inner brow raiser', 'Outer brow raiser', 'Brow lowerer','Cheek raiser', 'Lid tightener',
                   'Upper lip raiser','Lip corner puller','Dimpler','Lip corner depressor','Chin raiser',
                   'Lip tightener','Lip pressor']
    Ch_name_list2=['内眉提升', '外眉提升', '眉毛降低','脸颊提升','眼睑收紧','上最唇提升','嘴角向上','抿嘴',
                   '嘴角向下','下巴提升','嘴角拉紧','压嘴唇']
    label_pred[label_pred >= 0] = 1
    label_pred[label_pred < 0] = -1

    id = np.where(label_pred == 1)

    au_list = np.array(label_disfa)[id[1]]
    print('the predicted AU is ', au_list,id)
    names_list = []
    au_imgs = []
    k = len(au_list)
    plt.figure()
    for i in range(k):
        print('the num of au is ', au_list[i])
        print('the english au is ', En_name_list[id[1][i]])
        print('the english au is ', Ch_name_list[id[1][i]])
        names_list.append(str(au_list[i]) + '--' + str(En_name_list1[id[1][i]]) + '--' + str(Ch_name_list1[id[1][i]]))

        img = cv2.imread('d:/prj/ROI_AU/AU/' + 'AU' + str(au_list[i]) + '.jpg')
        img = cv2.resize(img, (176, 176))
        au_imgs.append(img)
        #plt.subplot(1, k, i + 1)
        #plt.imshow(img)




    return None
def show(data,landmarks):

    for i in range(66):
        cv2.circle(data, (int(landmarks[:, 2 * i]), int(landmarks[:, 2 * i + 1])), 2, (255, 255, 0))
    cv2.imwrite('landmark_test.jpg', data)



def predict():

    

    org_data = cv2.imread('au.jpg')

    org_data = cv2.resize(org_data, (176, 176))
    test_data = org_data[np.newaxis, :]
    time1=time.time()
    model = AU_detect()
    model.load_weights('compressed_JAA.h5',by_name=True)
    landmark,label_pred = model.predict(test_data)
    time2=time.time()
    print('the time used is ',time2-time1)

    label_list = [1, 2, 4, 6, 7, 9, 10, 12, 14, 15, 17, 23, 24, 25] ## total 14 labels
    label_BP4D=[1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 34]# not including 9 and 25,TOTAL 12 labels
    label_disfa = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]## total 12 labels
    label_paindata=[0, 4, 6, 7, 9, 10,  12, 15,  20, 25, 26, 27,43,50]##total 14 labels
    En_name_list = ['Inner brow raiser', 'Outer brow raiser', 'Brow lowerer', 'Upper lid raiser',
                    'Cheek raiser', 'Nose wrinkler', 'Lip corner puller', 'Lip corner depressor', 'Chin raiser', 'Lip part',
                    'Lip stretcher', 'Lips part', 'Jaw drop']
    Ch_name_list = ['内眉提升', '外眉提升', '眉毛降低','上睑提升', '脸颊提升',
                    '皱鼻', '嘴角向上', '嘴角向下', '下巴提升',
                    '嘴角拉伸', '嘴唇微张','颌部下降']

    label_pred[label_pred >= 0] = 1
    label_pred[label_pred < 0] = -1

    id = np.where(label_pred == 1)

    au_list = np.array(label_disfa)[id[1]]
    print('the predicted AU is ',au_list)
    names_list = []
    au_imgs = []
    k = len(au_list)
    plt.figure()
    for i in range(k):
        print('the num of au is ', au_list[i])
        print('the english au is ', En_name_list[id[1][i]])
        print('the english au is ', Ch_name_list[id[1][i]])
        names_list.append(str(au_list[i]) + '--' + str(En_name_list[id[1][i]]) + '--' + str(Ch_name_list[id[1][i]]))

        #img = cv2.imread('AU/' + 'AU' + str(au_list[i]) + '.jpg')
        #img = cv2.resize(img, (176, 176))
        #au_imgs.append(img)
        #plt.subplot(1, k, i+1)
        #plt.imshow('AU/' + 'AU' + au_list[i] + '.png')

    return landmark,names_list,au_imgs
def predict_video(video):
    from ROI_AU.roi_au import ROI_LSTM, vgg16, get_roi_patch
    videoCapture = cv2.VideoCapture(video)
    frames = []
    success, frame = videoCapture.read()
    frames.append(frame)
    k = 0
    while success:
        k += 1
        success, frame = videoCapture.read()
        frames.append(frame)

    test_data = frames[-1]
    test_data=cv2.resize(test_data,(200,200))
    test_patch = get_roi_patch(test_data)
    model = ROI_LSTM()
    model.load_weights('roi_lstm.npy')
    label_pred = model.predict(test_patch)

    label_list = [1, 2, 4, 6, 7, 9, 10, 12, 14, 15, 17, 23, 24, 25]
    En_name_list = ['Inner brow raiser', 'Outer brow raiser', 'Brow lowerer', 'Cheek raiser', 'Lid tightener',
                    'Nose wrinkler', 'Upper lip raiser', 'Lip corner puller', 'Dimpler', 'Lip corner depressor',
                    'Chin raiser', 'Lip tightener', 'Lip pressor', 'Lip part']
    Ch_name_list = ['抬起眉毛内角', '抬起眉毛外角', '皱眉（降低眉毛）', '脸颊提升', '眼轮匝肌内部收紧',
                    '皱鼻', '上嘴唇向上', '拉动嘴角', '收紧嘴角（抿嘴）', '嘴角向下',
                    '下唇向上', '收紧嘴唇', '嘴唇相互按压', '张嘴']

    label_pred[label_pred >= 0] = 1
    label_pred[label_pred < 0] = -1

    id = np.where(label_pred == 1)
    print(id)
    au_list = np.array(label_list)[id]
    print('the predicted AU is ',au_list)
    names_list = []
    au_imgs = []
    k = len(au_list)
    plt.figure()
    for i in range(k):
        print('the num of au is ', au_list[i])
        print('the english au is ', En_name_list[id[i]])
        print('the english au is ', Ch_name_list[id[i]])
        names_list.append(au_list[i] + '--' + En_name_list[id[0][i]] + '--' + Ch_name_list[id[0][i]])

        img = cv2.imread('AU/' + 'AU' + str(au_list[i]) + '.jpg')
        au_imgs.append(img)
        plt.subplot(1, k, i)
        #plt.imshow('AU/' + 'AU' + au_list[i] + '.png')

    return names_list,au_imgs


if __name__=='__main__':
    #test()
    predict()



