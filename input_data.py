import cv2
import os
import numpy as np
from skimage import transform
from utils import transform_data,transform_data_point2,transform_data_point,rotate,resize_data_point
import random





def get_data_labels_org():
    lines = open('d:\prj\ROI_AU\label_org.txt', 'r')
    lines = list(lines)
    data = []
    labels = []
    for ind in range(len(lines)):
        if lines[ind] != '\n':
            line = lines[ind].strip('\n').split(' ')
            data_dir = line[0]
            label = line[1:]
            data_name = 'd:/prj/ROI_AU' + '/' + data_dir

            img = cv2.imread(data_name)
            #img=transform.resize(img,(224,224,3))
            #img_data = np.array(img)

            data.append(img)
            labels.append(label)
    dataset=np.array(data)
    mlabels=np.array(labels)
    print('the shape of label is ', mlabels.shape)
    return dataset,mlabels

def pack_frames():
    ## to input sequence data to conv2d,we pack the 24 frames as channel
    base_dir='d:/prj/ROI_AU/images_seq'
    person=os.listdir(base_dir)
    persons=list(person)
    pack_data=[]
    for i in range(len(persons)):
        aperson=os.listdir(os.path.join(base_dir,persons[i]))
        aperson=list(aperson)
        org=cv2.imread(os.path.join(base_dir,persons[i],aperson[0]))
        for j in range(1,len(aperson)):
            tmp=cv2.imread(os.path.join(base_dir,persons[i],aperson[j]))
            tmp=np.concatenate((org,tmp),axis=2)
            org=tmp
        pack_data.append(tmp)
    pack_data=np.array(pack_data)
    print('the shape of pack data is ',pack_data.shape)
    return pack_data
def  get_data_labels_landmark():
    ## this is images_seq 24 images each video for paindata
    lines = open('/data/ai/DISFA/disfa_aus_both_landmark.txt', 'r')
    lines = list(lines)

    data = []
    labels = []
    names=[]
    landmarks=[]
    
    for ind in range(0,len(lines),30):

        line = lines[ind].strip('\n').split()
        data_dir = line[0]

        label = line[1:13]
        landmark=np.array(line[13:145])
        tmp=np.zeros((132,))
        for k in range(132):
            tmp[k]=float(landmark[k])
        
        data_name = '/data/ai/xxy/ROI_AU/' + data_dir
        img = cv2.imread(data_dir)
        h,w,c=img.shape
        if img is None:
            print('the dir of data is ',data_name)
        img=cv2.resize(img,(176,176))
        img_data = np.array(img)

        data.append(img)
        labels.append(label)
        names.append(data_dir)
        
        tmp[0::2]=tmp[0::2]*176/w
        tmp[1::2]=tmp[1::2]*176/h
        #show_landmark(img,tmp)
        landmarks.append(tmp)
    #labs=labels[::24]## get the label of each video
    #mlabels = np.array(labs)
    dataset=np.array(data)

    mut_labels=np.array(labels)
    landmarks=np.array(landmarks)
    loss_true=np.ones((dataset.shape[0],1))

    print('the shape of label is ',dataset.shape)
    return dataset,mut_labels,landmarks,loss_true

def  get_transformed_data_labels_landmark():
    ## this is images_seq 24 images each video for paindata
    lines = open('/data/ai/xxy/ROI_AU/Frame_Labels/paindata_landmark_labels_part2.txt', 'r')
    
    lines = list(lines)

    data = []
    labels = []
    names=[]
    landmarks=[]

    for ind in range(0,len(lines)):

        line = lines[ind].strip('\n').split()
        data_dir = line[0]

        label = line[1:15]
        landmark=np.array(line[15:147])
        tmp=np.zeros((132,))
        for k in range(132):
            tmp[k]=float(landmark[k])
        
        
        data_name = '/data/ai/xxy/ROI_AU/' + data_dir
        img = cv2.imread(data_dir)
        if img is None:
            print('the dir of data is ',data_name)
        
        img,tmp=transform_data_point(img,tmp)
        #print('the shape of img is',img.shape)

        data.append(img)
        labels.append(label)
        names.append(data_dir)       
        
        landmarks.append(tmp)
    #labs=labels[::24]## get the label of each video
    #mlabels = np.array(labs)
    dataset=np.array(data)

    mut_labels=np.array(labels)
    landmarks=np.array(landmarks)
    loss_true=np.ones((dataset.shape[0],1))

    print('the shape of label is ',dataset.shape)
    return dataset,mut_labels,landmarks,loss_true

def  get_ck_data_labels_landmark():
    ## this is images_seq 24 images each video
    lines = open('/data/ai/xxy/ROI_AU/JAA_Net/CK_emotion_au_landmark.txt', 'r')
    lines = list(lines)

    data = []
    labels = []
    names=[]
    landmarks=[]

    for ind in range(1):

        line = lines[ind].strip('\n').split()
        data_dir = line[0]

        label = line[2:16]
        landmark=np.array(line[16:152])

        tmp=np.zeros((136,))
        for k in range(136):
            tmp[k]=float(landmark[k])
        
        
        data_name = '/data/ai/xxy/ROI_AU/' + data_dir
        img = cv2.imread(data_name)
        
        if img is None:
            print('the dir of data is ',data_name)
        
        img,tmp=transform_data_point(img,tmp)
        #img,tmp,box=transform_data_point2(img,tmp)
        

        data.append(img)
        labels.append(label)
        names.append(data_dir)    
        
        landmarks.append(tmp)
    #labs=labels[::24]## get the label of each video
    #mlabels = np.array(labs)
    dataset=np.array(data)

    mut_labels=np.array(labels)
    landmarks=np.array(landmarks)
    loss_true=np.ones((dataset.shape[0],1))

    print('the shape of label is ',dataset.shape)
    return dataset,mut_labels,landmarks,loss_true

def  get_disfa_data_labels_landmark():
    ## this is images_seq 24 images each video
    lines = open('/data/ai/DISFA/disfa_aus_both_landmark.txt', 'r')
    lines = list(lines)
    num_data=0
    data = []
    labels = []
    
    landmarks=[]

    for ind in range(0,1):
        num_data +=1
        line = lines[ind].strip('\n').split()
        data_dir = line[0]

        label = line[1:13]
        landmark=np.array(line[13:145])
        tmp=np.zeros((132,))
        for k in range(132):
            tmp[k]=float(landmark[k])
        
        
        data_name = '/data/ai/xxy/ROI_AU/' + data_dir
        img = cv2.imread(data_dir)
        print('the org shape of img is',img.shape)
        if img is None:
            print('the dir of data is ',data_name)
        
        #img,points,box=transform_data_point2(img,tmp)
        img,points=resize_data_point(img,tmp)
        
       
        

        data.append(img)
        labels.append(label)       
        #print('the shape of points is ',points.shape)
        landmarks.append(points)
        #if random.choice([-1,0,1])>0:
         #   angle=random.choice([5,-5])
            
            #img_rot,points_rot=rotate(img,box,points,angle)
            
            #data.append(img_rot)
            #landmarks.append(points_rot)
            #labels.append(label)
            #num_data +=1

   
    dataset=np.array(data)

    mut_labels=np.array(labels)
    landmarks=np.array(landmarks)
    loss_true=np.ones((num_data,1))

    print('the shape of label is ',dataset.shape,landmarks.shape)
    return dataset,mut_labels,landmarks,loss_true

def show_landmark(img,landmark):
    print(landmark)
    for i in range(66):
        cv2.circle(img, (round(float(landmark[2*i])), round(float(landmark[2*i+1]))), 2, (0, 255, 0))
        cv2.imwrite('/data/ai/xxy/ROI_AU/JAA_Net/flipped.jpg', img)

if __name__=='__main__':
    get_data_labels_landmark()




