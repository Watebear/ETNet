import os
import numpy as np
import cv2


video_src_path = 'D:\project_jaanet\DISFA\Videos_LeftCamera'
label_name = os.listdir(video_src_path)
label_dir = {}
index = 0
picture_save_path = 'D:\project_jaanet\PyTorch-JAANet-master\data\imgs\DISFA'

for avi_name in label_name:
    label_dir[avi_name] = index
    index+=1
    video_save_path = avi_name[9:14]
    #print(video_save_path)
    each_video_save_full_path = os.path.join(picture_save_path,video_save_path)
    if not os.path.exists(each_video_save_full_path):
        os.mkdir(each_video_save_full_path)
    
    each_video_full_path = os.path.join(video_src_path,avi_name)
    cap = cv2.VideoCapture(each_video_full_path)
    frame_count = 0
    success = True
    while success:
        success,frame = cap.read()
        
        params=[]
        params.append(1)
        if success:
            img_ = os.path.join(each_video_save_full_path,'%d.jpg'%frame_count)
            cv2.imwrite(img_, frame,params)
        frame_count+=1
    cap.release()
    

    
    
    
    
