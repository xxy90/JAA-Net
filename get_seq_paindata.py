import os
import numpy as np
import cv2
from PIL import Image
from shutil import copyfile

num_each_person=23

labels=open('Frame_Labels/seq_labels.txt','w')
aus=[]
def get_seq_data_labels():
    persons=os.listdir('Frame_Labels/FACS')
    persons=list(persons)
    persons.sort()
    for i in range(len(persons)):
        aperson=[]
        
        print(os.path.join('Frame_Labels/FACS',persons[i]))
        aperson=os.listdir(os.path.join('Frame_Labels/FACS',persons[i]))       
        
        aperson = list(aperson)
        aperson.sort()
        for j in range(len(aperson)):
            print('persons[i] is ',persons[i])
            print('apersons[j] is ',aperson[j])
            print('------------- ')
            kk=0
            avideo=os.listdir(os.path.join('Frame_Labels/FACS',persons[i],aperson[j]))            
            #avideo=list(avideo)
            new_path=os.path.join('/data/ai/xxy/ROI_AU/Images_painseq', persons[i], aperson[j])
            aus_avideo = []
            for frame in avideo:

                frame_dir = os.path.join('Frame_Labels/FACS', persons[i], aperson[j], frame)
                size=os.path.getsize(frame_dir)

                if size!=0:
                    
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                        print('created ',new_path)
                    
                    
                    lines=open(frame_dir,'r')
                    lines=list(lines)
                    data_name = str(frame.split('_')[0]) + '.png'
                    data_name_new=str(frame.split('_')[0][:-3])+'%03d' % kk+'.png'
                    kk+=1
                    copyfile(os.path.join('Images' , persons[i], aperson[j])+'/'+data_name,
                             new_path+'/'+data_name_new)
                    #labels.write(new_path+data_name_new)
                    ## we just get the last label frame of a video
                    tmp_au=[]
                    for k in range(len(lines)):
                        num = lines[k].split()[0]
                        tmp_au.append(int(float(num)))
                        labels.write(' '+str(int(float(num))))
                    #labels.write('\n')
                    aus_avideo.append(tmp_au)
            ## align the frames to 24
            if os.path.exists( new_path):
                frames_new=os.listdir(new_path)
                frames_new=list(frames_new)
                while kk < num_each_person:
                    # np.random.shuffle(frames)
                    print('the image name to open is ',frames_new[-1])
                    #img = Image.open(new_path + '/' + str(frames_new[-1]))

                    # rot_img=img.rotate(180)
                    kk += 1
                    new_name = str(frames_new[-1].split('.')[0][:-3])  + '%03d' % kk + '.png'
                    #img.save(new_path + '/' + new_name, 'png')
                    copyfile(new_path + '/' + str(frames_new[-1]),new_path + '/' + new_name)
                    #frames_new.append(img)
                    aus_avideo.append(aus_avideo[-1])




                while kk > num_each_person:
                    # np.random.shuffle(frames)
                    #print('the current len(frames) is ', len(frames_new))

                    os.remove(new_path + '/' + frames_new[0])
                    del frames_new[0]
                    del aus_avideo[0]
                    kk -= 1

                ## after align write down labels
                after_align=os.listdir(new_path)
                after_align=list(after_align)
                for ii in range(len(after_align)):
                    label_str = [' ' + str(label) for label in aus_avideo[ii]]
                    labels.write(new_path+'/'+after_align[ii])
                    labels.write(''.join(label_str))
                    labels.write('\n')


    labels.close()


    return aus

def make_multi_label():
    #aus=get_labels()
    aus=[0, 4, 6, 7, 9, 10,  12, 15,  20, 25, 26, 27,43,50]## 14 labels
    aus.sort()
    print(aus)
    label_list=list(set(aus))
    f=open('Frame_Labels/FACS/labels.txt','r')
    multi_labels=open('Frame_Labels/FACS/labels.txt','w')
    #f=list(f)

    for li in (f):
        base_label = [-1] * len(label_list)
        line=li.strip('\n').split()
        tmp_label=line[1:]
        data_name=line[0]
        data_name.replace('Frame_Labels/FACS','Images')
        name=data_name.split('_')[0]+'.png'
        multi_labels.write(name+' ')

        for label in tmp_label:
            print(' label is ',label)
            for j in range(len(label_list)):
                if int(label)==label_list[j]:
                    base_label[j]=1
        print(base_label)
        str_label = [' '+str(k)  for k in base_label]
        multi_labels.write(''.join(str_label) + '\n')
    multi_labels.close()

    return multi_labels


if __name__=='__main__':

    get_seq_data_labels()







