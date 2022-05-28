#!/usr/bin/env python
#type:ignore
import sys
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospy
import numpy as np
import time
import message_filters
import os
import shutil 

class image_convert:
    def __init__(self):
        self.image_depth=message_filters.Subscriber("/camera/depth/image_raw",Image)
        self.image_rgb=message_filters.Subscriber("/camera/rgb/image_raw",Image)
        self.bridge=CvBridge()
        self.time_sychronization=message_filters.ApproximateTimeSynchronizer([self.image_depth,self.image_rgb],queue_size=10,slop=0.01,allow_headerless=True)
        self.start_time=time.time()

    def callback(self,image_depth,image_rgb):
        cv_image_rgb=self.bridge.imgmsg_to_cv2(image_rgb)
        cv_image_rgb=cv2.cvtColor(cv_image_rgb, cv2.COLOR_BGR2RGB)
        cv_image_depth=self.bridge.imgmsg_to_cv2(image_depth,"32FC1")
        cv_image_depth = np.array(cv_image_depth, dtype=np.float32)
        cv2.normalize(cv_image_depth, cv_image_depth, 0, 1, cv2.NORM_MINMAX)
        cv_image_depth=cv_image_depth*255
        cv_image_depth_real=self.bridge.imgmsg_to_cv2(image_depth,"16UC1")
        max_meter=3
        cv_image_depth_real=np.array(cv_image_depth_real/max_meter,dtype=np.uint8)
        depth_image_original=cv_image_depth_real
        image=cv_image_depth
        mask=np.ones(image.shape)*255
        for i in range(len(image)):
            for j in range(len(image[i])):
                    if 10<image[i][j]<50:
                        if 130<j<470 and i>80:
                            mask[i][j]=0
        masked_image=cv_image_depth_real
        masked_image[mask>0]=0
        if time.time()-self.start_time>2:
            cv2.imwrite('/home/kentuen/ros_ws/src/kcnet_garnet_project/src/kcnet_original_images/%f_image.png'%(time.time()),masked_image)
            cv2.imwrite('/home/kentuen/ros_ws/src/kcnet_garnet_project/src/kcnet_original_images/%f_mask.png'%(time.time()),mask)
            cv2.imwrite('/home/kentuen/ros_ws/src/kcnet_garnet_project/src/kcnet_original_images/%f_rgb.png'%(time.time()),cv_image_rgb)
            print ('Photo taken!')
            self.start_time=time.time()
        cv2.waitKey(3)
    
    def image_capture(self):
        print ('image capture starts...')
        self.time_sychronization.registerCallback(self.callback)

def main(args):
    direcorty='/home/kentuen/ros_ws/src/kcnet_garnet_project/src/kcnet_original_images/'
    if os.path.exists(direcorty):
        print ("There exits a folder for orevious kcnet_original_images, deleting it...")
        shutil.rmtree(direcorty)
    if not os.path.exists(direcorty):
        print ("Constructing a new folder for a new kcnet_original_images...")
        os.makedirs(direcorty)
    rospy.init_node("cv_image_convertor",anonymous=True)
    convertor=image_convert()
    convertor.image_capture()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shut Down...")

if __name__=="__main__":
    main(sys.argv)
