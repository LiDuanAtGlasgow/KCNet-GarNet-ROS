#type:ignore
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import cv2
import numpy as np
import glob
import time

depth_folder='./depth_images/'
rgb_folder='./rgb_images/'
masked_depth_folder='./masked_depth/'
masked_rgb_folder='./masked_rgb/'
if not os.path.exists(depth_folder):
    os.makedirs(depth_folder)
if not os.path.exists(rgb_folder):
    os.makedirs(rgb_folder)
if not os.path.exists(masked_depth_folder):
    os.makedirs(masked_depth_folder)
if not os.path.exists(masked_rgb_folder):
    os.makedirs(masked_rgb_folder)

i=0
for fil_name in sorted(glob.glob('./images/*_depth.png'),key=str.lower):
    image=cv2.imread(fil_name)
    image_path=depth_folder+str(i+1).zfill(4)+'.png'
    i+=1
    cv2.imwrite(image_path,image)

i=0
for fil_name in sorted(glob.glob('./images/*_rgb.png'),key=str.lower):
    image=cv2.imread(fil_name)
    image_path=rgb_folder+str(i+1).zfill(4)+'.png'
    i+=1
    cv2.imwrite(image_path,image)

start_time=time.time()
num_images=len(next(os.walk(depth_folder))[2])
for idx in range(num_images):
    image_path=depth_folder+str(idx+1).zfill(4)+'.png'
    image=cv2.imread(image_path,0)
    mask=np.ones(image.shape)*255
    for i in range(len(image)):
        if idx<=15:
            for j in range(len(image[i])):
                if 0<image[i][j]<55:
                    if 100<j<470 and 300>i>270:
                        mask[i][j]=0
        if 15<idx:
            for j in range(len(image[i])):
                if 0<image[i][j]<55:
                    if 100<j<470 and 300>i>270-int(((250/45)*(idx-14))):
                        mask[i][j]=0

    rgb_mask=np.ones(image.shape)*255
    shift_step=10
    for i in range (len(rgb_mask)):
        for j in range (len(rgb_mask[i])-shift_step):
            rgb_mask[i][j]=mask[i][j+shift_step]

    depth_image=cv2.imread(depth_folder+str(idx+1).zfill(4)+'.png')
    depth_image[mask>0]=0
    cv2.imwrite(masked_depth_folder+str(idx+1).zfill(4)+'.png',depth_image)
    rgb_image=cv2.imread(rgb_folder+str(idx+1).zfill(4)+'.png')
    rgb_image[rgb_mask>0]=0
    cv2.imwrite(masked_rgb_folder+str(idx+1).zfill(4)+'.png',rgb_image)
    if idx%int(num_images/10)==0:
        print('No',idx+1,'has been finished! time consumed:',time.time()-start_time)
        start_time=time.time()
print ('finished!')