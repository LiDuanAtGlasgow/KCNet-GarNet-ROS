#type:ignore
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import cv2
import numpy as np

image_path='./depth.png'
image=cv2.imread(image_path,0)
mask=np.ones(image.shape)*255

for i in range(len(image)):
    for j in range(len(image[i])):
        if 0<image[i][j]<70:
            if 130<j<470 and i>80:
                mask[i][j]=0

rgb_mask=np.ones(image.shape)*255
shift_step=10
for i in range (len(rgb_mask)):
    for j in range (len(rgb_mask[i])-shift_step):
        rgb_mask[i][j]=mask[i][j+shift_step]

cv2.imwrite('./mask.png',mask)
cv2.imwrite('./rgb_mask.png',rgb_mask)
depth_image=cv2.imread('./depth.png')
depth_image[mask>0]=0
cv2.imwrite('./masked_depth.png',depth_image)
rgb_image=cv2.imread('./rgb.png')
rgb_image[rgb_mask>0]=0
cv2.imwrite('./masked_rgb.png',rgb_image)
print ('finished!')