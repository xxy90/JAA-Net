import os
import numpy as np

persons=os.listdir('Frame_Labels/FACS')
persons=list(persons)

aus=[]
def get_labels():
    labels = open('Frame_Labels/FACS/labels.txt', 'w')
    for i in range(len(persons)):
        aperson=os.listdir(os.path.join('Frame_Labels/FACS',persons[i]))
        aperson = list(aperson)
        for j in range(len(aperson)):
            avideo=os.listdir(os.path.join('Frame_Labels/FACS',persons[i],aperson[j]))
            #avideo=list(avideo)
            for frame in avideo:
                frame_dir = os.path.join('Frame_Labels/FACS', persons[i], aperson[j], frame)
                size=os.path.getsize(frame_dir)
                if size!=0:
                    labels.write(frame_dir+' ')
                    lines=open(frame_dir,'r')
                    lines=list(lines)

                    for k in range(len(lines)):
                        num=lines[k].split()[1]
                        aus.append(int(float(num)))
                        if k!= len(lines)-1:
                            labels.write(str(int(float(num))) + ' ')
                        else:
                            labels.write(str(int(float(num))) + '\n')
    labels.close()


    return aus

def make_multi_label():
    #aus=get_labels()
    aus=[0, 4, 6, 7, 9, 10,  12, 15,  20, 25, 26, 27,43,50]## 14 labels
    aus.sort()
    print(aus)
    label_list=list(set(aus))
    f=open('Frame_Labels/labels.txt','r')
    multi_labels=open('Frame_Labels/landmark_labels.txt','w')
    #f=list(f)

    for li in (f):
        base_label = [-1] * len(label_list)
        line=li.strip('\n').split()
        tmp_label=line[1:]
        data_name=line[0]

        name=data_name.split('_facs.txt')[0]+'.png'
        name.replace('Frame_Labels/FACS', 'Images')
        multi_labels.write(name+' ')

        for label in tmp_label:
            print(' label is ',label)
            for j in range(len(label_list)):
                if int(label)==label_list[j]:
                    base_label[j]=1
        print(base_label)
        str_label = [' '+str(k)  for k in base_label]
        multi_labels.write(''.join(str_label) )

        landmark_names=(str(data_name.split('_facs.txt'))[0]+'_aam.txt').replace('FACS','AAM_landmarks')
        landmark_file=open(landmark_names,'r')
        lines=list(landmark_file)
        landmarks=[]
        for i in range (len(lines)):
            line=lines[i].strip('\n').split()
            landmarks.append(line[0])
            landmarks.append(line[1])
        str_landmark=[' '+str(kk) for kk in landmarks]
        multi_labels.write(''.join(str_landmark)+ '\n')

    multi_labels.close()

    return multi_labels


if __name__=='__main__':

    make_multi_label()







