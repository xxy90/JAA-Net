## Description
This is keras implemtation for JAA-NET:Deep Adaptive Attention for Joint Facial Action
Unit Detection and Face Alignment

## For Training
Use the command python train_model.py
*AU_detect model did not contains attention block

*AU_detect2 model contains attention block,and the init weight for attention is propotional to distance of landmark points

*AU_detect2 model contains attention block,and the init weight for attention is implementation of paper

## Result

landmarks and Multilabel AUs for person,i.e if the AUs base dataset is BP4D,

the corresponding AU number is  label_disfa = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26].

For example , the AUs result is [0,1,1,0,0,0,0,0,0,0,0,1],the people have AUs[2,4,26]

## predict 

run python test_JAA.py




##the labels introductions
You can also make datasets yourself ,the labels contains image_id and AUS and 68 landmarks

For CK+ dataset

label.txt is the CK+ dataset with only 936 images selected from label_org.txt

label_org is the CK+ dataset with only 2250 images

label_line.txt is the CK+ dataset  with labels only choosing the first data of one line

label_v.txt is the unique prefix of CK+ dataset and to be used for labels_per_video.txt

labels_per_video.txt is the sequence of CK+ dataset ,in which the frames is 24 for each video