import numpy as np

import tensorflow as tf
from keras.layers import Lambda
import keras.backend as K
from face_align import face_landmark
from pre_learing import multi_scale_learning
import dlib
import cv2

target_size=176

def transform_data_point(data,points):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    rects = detector(data, 1)
    box=np.zeros((4,))
    height, width, ch = data.shape
    if len(rects)==0:
        d=dlib.rectangle(0,0,width,height)
    else:
        d=rects[0]
    box[0]=d.left()
    box[1]=d.top()
    box[2]=d.right()
    box[3]=d.bottom()

    square_box=get_square_box(box)
    w = square_box[2] - square_box[0]
    scale_ratio = 176 / w
    if scale_ratio>=1:
        #print('expanding data')
        box=expand_box(square_box,scale_ratio)
    else:
        print('downsizing data')
        box=downsizing_box(square_box,scale_ratio)
    if box[2] > width:
        box[0] = box[0] - (box[2] - width)
    if box[3] > height:
        box[1] = box[1] - (box[3] - height)
    data=data[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:]
    for i in range(66):
        points[2*i]-=box[0]
        points[2*i+1]-=box[1]
    return data,points

def resize_data_point(data,points):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    rects = detector(data, 1)
    box=np.zeros((4,))
    height, width, ch = data.shape
    if len(rects)==0:
        d=dlib.rectangle(0,0,width,height)
    else:
        d=rects[0]
    box[0]=d.left()
    box[1]=d.top()
    box[2]=d.right()
    box[3]=d.bottom()

    square_box=get_square_box(box)
    w = square_box[2] - square_box[0]
    scale_ratio = target_size / w
    if scale_ratio>=1:
        #print('expanding data')
        box=expand_box(square_box,scale_ratio)
        if box[2] > width:
            box[0] = box[0] - (box[2] - width)
        if box[3] > height:
            box[1] = box[1] - (box[3] - height)
        data = data[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
        for i in range(66):
            points[2 * i] -= box[0]
            points[2 * i + 1] -= box[1]
    else:
        print('downsizing data')
        #box=downsizing_box(square_box,scale_ratio)

        data = data[int(square_box[1]):int(square_box[3]), int(square_box[0]):int(square_box[2]), :]
        for i in range(66):
            points[2 * i] -= square_box[0]
            points[2 * i + 1] -= square_box[1]
        data=cv2.resize(data,(target_size,target_size))
        points =points*scale_ratio

    return data,points


def transform_data(data):
    ## use dlib to get face box and then padding to 176*176
    # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    rects = detector(data, 1)
    height, width, ch = data.shape
    box = np.zeros((4,))
    if len(rects) == 0:
        d = dlib.rectangle(0, 0, width, height)
    else:
        d = rects[0]
    box[0] = d.left()
    box[1] = d.top()
    box[2] = d.right()
    box[3] = d.bottom()

    square_box = get_square_box(box)
    w = square_box[2] - square_box[0]
    scale_ratio = 176 / w
    if scale_ratio >= 1:
        print('expanding the box')
        box_n = expand_box(square_box, scale_ratio)
    else:
        print('downsizing the box')
        box_n = downsizing_box(square_box, scale_ratio)
    # print('the box is ',int(box_n[1]),int(box_n[3]))
    if box_n[2] > width:
        box_n[0] = box_n[0] - (box_n[2] - width)
    if box_n[3] > height:
        box_n[1] = box_n[1] - (box_n[3] - height)
    data = data[int(box_n[1]):int(box_n[3]), int(box_n[0]):int(box_n[2]), :]
    print('the shape of org_data is', data.shape)
    return data
def transform_data_point2(data,points):
    box=get_minimal_box(points)
    square_box=get_square_box(box)
    w = square_box[2] - square_box[0]
    scale_ratio = 176 / w
    height, width, ch = data.shape
    if scale_ratio>=1:
        print('expanding box')
        box=expand_box(square_box,scale_ratio)
    else:
        #print('downsizing box')
        box=downsizing_box(square_box,scale_ratio)
    if box[2] > width:
        box[0] = box[0] - (box[2] - width)
    if box[3] > height:
        box[1] = box[1] - (box[3] - height)
    data=data[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:]
    for i in range(66):
        points[2*i]-=box[0]
        points[2*i+1]-=box[1]
    return data,points,box
def get_minimal_box(points):
    """
    Get the minimal bounding box of a group of points.
    The coordinates are also converted to int numbers.
    """
    min_x = int(min([points[2*i] for i in range(66)]))
    max_x = int(max([points[2*i] for i in range(66)]))
    min_y = int(min([points[2*i+1] for i in range(66)]))
    max_y = int(max([points[2*i+1] for i in range(66)]))
    return [min_x, min_y, max_x, max_y]


def expand_box(square_box, scale_ratio):
    """Scale up the box"""
    assert (scale_ratio >= 1), "Scale ratio should be greater than 1."
    delta = int((square_box[2] - square_box[0]) * (scale_ratio - 1) / 2)
    left_x = max(square_box[0] - delta,0)
    left_y = max(square_box[1] - delta,0)
    right_x = square_box[2] + delta-min(square_box[0] - delta,0)
    right_y = square_box[3] + delta-min(square_box[1] - delta,0)
    dif=right_x-left_x-176
    dif2=right_y-left_y-176
    if dif!=0:
        right_x -=dif
    if dif2!=0:
        right_y -=dif2

    return [left_x, left_y, right_x, right_y]

def downsizing_box(square_box, scale_ratio):
    delta = int((square_box[2] - square_box[0]) * (1-scale_ratio) / 2)
    left_x = square_box[0] + delta
    left_y = square_box[1] + delta
    right_x = square_box[2] - delta
    right_y = square_box[3] - delta
    dif = right_x - left_x - 176
    dif2 = right_y - left_y - 176
    if dif != 0:
        right_x -= dif
    if dif2 != 0:
        right_y -= dif2

    return [left_x, left_y, right_x, right_y]

    




def get_square_box(box):
    """Get the square boxes which are ready for CNN from the boxes"""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:                   # Already a square.
        return box
    elif diff > 0:                  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:                           # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    # Make sure box is always square.
    assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

    return [left_x, top_y, right_x, bottom_y]





def au_loss(label,pred):
    ## the shape of pred and label is (num_data,12)
    loss=0
    R=get_rate_for_label(label)
    pred = tf.div(pred + 1, 2)
    label = tf.convert_to_tensor(label)
    shape = label.get_shape().as_list()
    for i in range(12):
        temp_loss=-R[i]*(label[:,i]*tf.log((pred[:,i]+0.05)/1.05)+(1-label[:,i])*tf.log((1.05-pred[:,i])/1.05))
        temp_sum=tf.reduce_mean(temp_loss)
        loss+=temp_sum
    f1=micro_f1(label,pred)
    loss2=tf.reduce_mean(tf.multiply(R,1-f1))
    loss=loss+loss2
    return loss
def micro_f1(label,pred):
    ##pred and label need to be tensor,pred is (-1,1)

    ## get the number of data_size and labels
    num_labels=14

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
    def recall(y_true, y_pred):
        """Recall metric.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        print('recall shape is ', recall.get_shape().as_list())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        print('precision shape is ',precision.get_shape().as_list())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def get_w(shape):
    init_w = np.zeros((16,11,11,48))  # num_data,16,11,11,48
    # print('the shape of init_w is ',init_w.shape)
    w = []
    for j in range(-5, 6):
        for k in range(-5, 6):
            init_w[:, j, k, :]=(init_w[:, j, k, :] + abs(j) + abs(k)) / 11

    w = K.variable(w)
    return w

def get_w2(point):
    ## the shape of w is [num_data,16,176,176,3]
    pp=get_roi_patch(point)
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
        init_w=tf.shape(init_w,(-1,176,176))
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

    print('the shape of pp is ', pp.shape)
    init_w=[]
    for i in range(176):
        for j in range(176):
            tmp=1-tf.reduce_max((tf.abs(pp[:,:,0]-i)+tf.abs(pp[:,:,1]-j)),axis=1)/176
            zeros=tf.zeros_like(tmp)
            init_w.append(tf.reduce_max((tmp,zeros),axis=0))

    w = tf.convert_to_tensor(init_w)
    print('the shape of w is ',w.shape)
    w = tf.transpose(w, (1, 0))

    w = tf.reshape(w, (-1, 176, 176))

    w = tf.expand_dims(w, axis=3)
    w = tf.tile(w, (1, 1, 1, 3))

    return w

def get_au_point(point):
    ## data is tensors[num_data,88,88,48]
    #point,x=face_landmark(data)


    d= tf.sqrt(tf.square(point[:,42*2] - point[:,40*2])+tf.square(point[:,42*2+1] - point[:,40*2+1]))
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


    pp=tf.convert_to_tensor(pp)## shape is [17*2,num_data]
    pp=tf.transpose(pp,(1,0))## shape is [num_data,17*2]


    return pp

def get_roi_patch(img_input):
    ## data is tensors[num_data,88,88,48]
    #point,x=face_landmark(data)
    pre_fea = multi_scale_learning(img_input)

    point, data = face_landmark(pre_fea)

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


def rotate(img, bbox, landmark, alpha):
    """
        given a face with bbox and landmark, rotate with alpha
        and return rotated face with bbox, landmark (absolute position)
   
    """
    landmark=landmark.reshape(66,2)
    center = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    #whole image rotate
    #pay attention: 3rd param(col*row)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat,(img.shape[1],img.shape[0]))
    landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark])
    #crop face 
    face = img_rotated_by_alpha[bbox[1]:bbox[3]+1,bbox[0]:bbox[1]+1]
    landmark_=landmark_.reshape(132)
    return img_rotated_by_alpha, landmark_
