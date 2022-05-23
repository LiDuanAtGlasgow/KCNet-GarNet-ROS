#type:ignore
from calendar import c
import torch
import pandas as pd
import cv2
from torch.utils.data import Dataset
from typing import Any,Callable,Dict,IO,Optional,Tuple,Union
import os
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import csv
from Continuous_Perception import early_stop
import rospy
from moveit_python import PlanningSceneInterface, MoveGroupInterface
from geometry_msgs.msg import PoseStamped
import baxter_interface
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import PointCloud2
import rospy
import pcl
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import rospy
import ros_numpy
import csv
import sys

f=open('./point_clouds.csv','w')
csv_writer=csv.writer()
csv_writer.writerow(('no','x','y','z'))

np.random.seed(42)
torch.manual_seed(42)

cuda=torch.cuda.is_available()
model_path='./Model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
fig_path='./figures/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

physnet_classes=['11','12','13','14','15','16','17','18','19','20']
colors=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd','#b4b4be','#ff7f01','#2ca02c','#d62728','#9467bd']
numbers=[11,12,13,14,15,16,17,18,19,20]
mean,std=0.00586554,0.03234654
print ('physnet_claasses:',len(physnet_classes))
print ('colors:',len(colors))

def extract_embeddings(dataloader,model):
    with torch.no_grad():
        model.eval()
        embeddings=np.zeros((len(dataloader.dataset),2))
        labels=np.zeros(len(dataloader.dataset))
        video_labels=np.zeros(len(dataloader.dataset))
        k=0
        for images, target, video_label in dataloader:
            if cuda:
                images=images.cuda()
            embeddings[k:k+len(images)]=model.get_emdding(images).data.cpu().numpy()
            labels[k:k+len(images)]=target.numpy()
            video_labels[k:k+len(images)]=video_label.numpy()
            k+=len(images)
    return embeddings,labels,video_labels

class AlexNet_Embedding(nn.Module):
    def __init__(self):
        super(AlexNet_Embedding,self).__init__()
        self.model=frozon(models.alexnet(pretrained=True))
        self.model.features[0]=nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        self.model.classifier=nn.Sequential(
            nn.Linear(256*6*6,256),
            nn.PReLU(),
            nn.Linear(256,256),
            nn.PReLU(),
            nn.Linear(256,2)
        )
    
    def forward(self,x):
        output=self.model(x)
        return output
    
    def get_emdding(self,x):
        return self.forward(x)

class ResNet18_Embedding(nn.Module):
    def __init__(self) -> None:
        super(ResNet18_Embedding,self).__init__()
        modeling=frozon(models.resnet18(pretrained=True))
        modules=list(modeling.children())[:-2]
        self.features=nn.Sequential(*modules)
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc=nn.Sequential(
            nn.Linear(512*8*8,256),
            nn.PReLU(),
            nn.Linear(256,256),
            nn.PReLU(),
            nn.Linear(256,2)
        )

    def forward(self,x):
        output=self.features(x)
        output=output.reshape(output.shape[0],-1)
        output=self.fc(output)
        return output
    
    def get_emdding(self,x):
        return self.forward(x)

class TripletNet(nn.Module):
    def __init__(self,embedding_net):
        super(TripletNet,self).__init__()
        self.embedding_net=embedding_net
    
    def forward(self,x1,x2,x3):
        output1=self.embedding_net(x1)
        output2=self.embedding_net(x2)
        output3=self.embedding_net(x3)
        return output1,output2,output3
    
    def get_emdding(self,x):
        return self.embedding_net(x)

class GarNet_Dataset(Dataset):
    def __init__(self,file_path,csv_path,opt=1,transform:Optional[Callable]=None,target_transform:Optional[Callable]=None)->None:
        super(GarNet_Dataset,self).__init__()
        self.imgs_path=file_path
        self.csv_path=csv_path
        data=pd.read_csv(self.csv_path)
        self.labels=data.iloc[:,opt]
        self.transform=transform
        self.data=data.iloc[:,0]
        self.video_labels=data.iloc[:,3]
    
    def __getitem__(self,index:int)->Tuple[Any,Any]:
        imgs_path=self.imgs_path+self.data[index]
        target=int(self.labels[index])
        img=cv2.imread(imgs_path,0)
        img=Image.fromarray(img,mode='L')
        if self.transform is not None:
            img=self.transform(img)
        video_label=int(self.video_labels[index])
        return img, target, video_label
    
    def __len__(self):
        return len(self.data)

def standard_label(embeddings,targets,numbers=10):
    f=open('./standard_labels.csv','w')
    csv_writer=csv.writer(f)
    csv_writer.writerow(('Name','mean_x','mean_y','std_x','std_y'))
    for i in range(numbers):
        inds=np.where(targets==i+1)[0]
        embedding_x=embeddings[inds,0].mean()
        embedding_y=embeddings[inds,1].mean()
        embedding_stdx=embeddings[inds,0].std()
        embedding_stdy=embeddings[inds,1].std()
        csv_writer.writerow((i+1,embedding_x,embedding_y,embedding_stdx,embedding_stdy))

def plot_embeddings(embeddings,targets,xlim=None,ylim=None):
    plt.figure(figsize=(10,10))
    for i in range (len(physnet_classes)):
        inds=np.where(targets==numbers[i])[0]
        plt.scatter(embeddings[inds,0],embeddings[inds,1],alpha=0.5,color=colors[i])
    if xlim:
        plt.xlim(xlim[0],xlim[1])
    if ylim:
        plt.ylim(ylim[0],ylim[1])
    plt.legend(physnet_classes)
    plt.savefig(fig_path+'{:f}.png'.format(time.time()))
    plt.show()

def garnet_cp_plotting(cp_data,samp_number=600,number_category=5):
    samps_number=np.array([0,600,1800,3000])
    for i in range (number_category):
        if number_category==5:
            names=['Pant','Shirt','Sweater','Towel','T-shirt']
            df=pd.DataFrame({'x':cp_data[i*samp_number:(i+1)*samp_number,0],'pant':cp_data[i*samp_number:(i+1)*samp_number,1],
            'shirt':cp_data[i*samp_number:(i+1)*samp_number,2],'sweater':cp_data[i*samp_number:(i+1)*samp_number,3],
            'towel':cp_data[i*samp_number:(i+1)*samp_number,4],'tshirt':cp_data[i*samp_number:(i+1)*samp_number,5],
            'acc':cp_data[i*samp_number:(i+1)*samp_number,6]})
            plt.figure()
            subplot=plt.subplot()
            subplot.plot('x','pant',data=df,color='red',label='pant')
            subplot.plot('x','shirt',data=df,color='blue',label='shirt')
            subplot.plot('x','sweater',data=df,color='green',label='sweater')
            subplot.plot('x','towel',data=df,color='purple',label='towel')
            subplot.plot('x','tshirt',data=df,color='gray',label='tshirt')
            subplot.set_xlabel('Input')
            subplot.set_ylabel('Decision Distance')
            subplot2=subplot.twinx()
            subplot2.plot('x','acc',data=df,color='grey',linewidth=10,label='Accuracy',linestyle='dotted')
            subplot2.set_ylabel('Accuracy(%)')
            subplot2.set_ylim([0,100])
            plt.legend()
            plt.title(names[i])
            plt.savefig('./'+names[i]+'_category.png')
            plt.show()
        if number_category==3:
            names=['Light','Medium','Heavy']
            df=pd.DataFrame({'x':cp_data[samps_number[i]:samps_number[i+1],0],'light':cp_data[samps_number[i]:samps_number[i+1],1],
            'medium':cp_data[samps_number[i]:samps_number[i+1],2],'heavy':cp_data[samps_number[i]:samps_number[i+1],3],
            'acc':cp_data[samps_number[i]:samps_number[i+1],4]})
            plt.figure()
            subplot=plt.subplot()
            subplot.plot('x','light',data=df,color='red',label='light')
            subplot.plot('x','medium',data=df,color='blue',label='medium')
            subplot.plot('x','heavy',data=df,color='green',label='heavy')
            subplot.set_xlabel('Input')
            subplot.set_ylabel('Decision Distance')
            subplot2=subplot.twinx()
            subplot2.plot('x','acc',data=df,color='grey',linewidth=10,label='Accuracy',linestyle='dotted')
            subplot2.set_ylabel('Accuracy(%)')
            subplot2.set_ylim([0,100])
            plt.legend()
            plt.title(names[i])
            plt.savefig('./'+names[i]+'_category.png')
            plt.show()

def GarNet2KCNet():
    p = PlanningSceneInterface("base")
    g = MoveGroupInterface("both_arms", "base")
    gr = MoveGroupInterface("right_arm", "base")
    rightgripper=baxter_interface.Gripper('right')
    jts_both = ['left_e0', 'left_e1', 'left_s0', 'left_s1', 'left_w0', 'left_w1', 'left_w2', 'right_e0', 'right_e1', 'right_s0', 'right_s1', 'right_w0', 'right_w1', 'right_w2']
    pos1 = [-1.441426162661994, 0.8389151064712133, 0.14240920034028015, -0.14501001475655606, -1.7630090377446503, -1.5706376573674472, 0.09225918246029519,1.7238109084167481, 1.7169079948791506, 0.36930587426147465, -0.33249033539428713, -1.2160632682067871, 1.668587600115967, -1.810097327636719]
    g.moveToJointPosition(jts_both, pos1, plan_only=False)

    start_time=time.time()
    with open ('data_collection.csv','rb') as csvfile:
        reader=csv.DictReader(csvfile)
        n=0
        for row in reader:
            n+=1
        data=np.ones((n,9))
    with open ('data_collection.csv','rb') as csvfile:
        reader=csv.DictReader(csvfile)
        m=0
        for row in reader:
            data[m,0]=int(row['step'])
            data[m,1]=float(row['position_x'])
            data[m,2]=float(row['position_y'])
            data[m,3]=float(row['position_z'])
            data[m,4]=float(row['orientation_x'])
            data[m,5]=float(row['orientation_y'])
            data[m,6]=float(row['orientation_z'])
            data[m,7]=float(row['orientation_w'])
            data[m,8]=int(row['grippers'])
            m+=1
    print ('step len:',n)
    col_len=1
    n_epochs=1

    for epoch in range (n_epochs):
        for step in range (col_len):
            step+=1
            p.waitForSync()        
            pickgoal = PoseStamped() 
            pickgoal.header.frame_id = "base"
            pickgoal.header.stamp = rospy.Time.now()
            pickgoal.pose.position.x = data[step,1]
            pickgoal.pose.position.y = data[step,2]
            pickgoal.pose.position.z = data[step,3]
            pickgoal.pose.orientation.x = data[step,4]
            pickgoal.pose.orientation.y = data[step,5]
            pickgoal.pose.orientation.z = data[step,6]
            pickgoal.pose.orientation.w = data[step,7]
            gr.moveToPose(pickgoal, "right_gripper", plan_only=False)
            rospy.sleep(2.0)
            if data[step,8]==0:
                rightgripper.close()
            else:
                rightgripper.open()
            print ('step',step+1,'finished, time:',time.time()-start_time)
            start_time=time.time()

class Get_Images():
    def __init__(self,image,shape,transforms=None):
        self.image=image
        self.transform=transforms
        self.shape=shape
    
    def __getitem__(self):
        image=self.image
        if not self.transform == None:
            image=self.transform(image)
        image=torch.unsqueeze(image,dim=0)
        shape=enc.transform([[self.shape]]).toarray()
        shape=shape.astype(int)

        return image,shape
    
    def __len__(self):
        return len(self.image)

def frozon (model):
    for param in model.parameters():
        param.requires_grad=False
    return model

cclass KCNet(nn.Module):
    def __init__(self) -> None:
        super(KCNet,self).__init__()
        modeling=frozon(models.resnet18(pretrained=True))
        modules=list(modeling.children())[:-2]
        self.features=nn.Sequential(*modules)
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc0=nn.Sequential(
            nn.Linear(512*8*8,256),
            nn.PReLU()
        )
        self.fc1=nn.Sequential(
            nn.Linear(332,332),
            nn.PReLU()
        )
        self.fc2=nn.Sequential(
            nn.Linear(5,76),
            nn.PReLU()
        )
        self.fc3=nn.Sequential(
            nn.Linear(332,50),
            nn.PReLU()
        )

    def forward(self,x,shape):
        output=self.features(x)
        output=self.fc0(output.reshape(output.shape[0],-1))
        shape=self.fc2(shape)
        shape=shape.reshape(shape.shape[0],-1)
        output=torch.cat([output,shape],dim=1)
        output=self.fc3(self.fc1(output))
        output = F.log_softmax(output, dim=1)
        return output
    
    def get_emdding(self,x):
        return self.forward(x)

CATEGORIES=['towel','tshirt','shirt','sweater','jean']

def test(kcnet,data,true_label,correct,acc,category,position_index):
    output=kcnet(data)
    pred=output.argmax(dim=1,keepdim=True)
    print ('true_postion:',category,position_index)
    print('predicted_postion:',CATEGORIES[pred.item()//10],pred.item()%10)
    if true_label==pred.item():
        correct+=1
    acc+=1
    return pred,acc,correct

def manipulation(path):
    p = PlanningSceneInterface("base")
    g = MoveGroupInterface("both_arms", "base")
    gr = MoveGroupInterface("right_arm", "base")
    gl = MoveGroupInterface("left_arm", "base")
    leftgripper = baxter_interface.Gripper('left')
    rightgripper=baxter_interface.Gripper('right')

    name=path
    start_time=time.time()
    with open (name,'rb') as csvfile:
        reader=csv.DictReader(csvfile)
        n=0
        for row in reader:
            n+=1
        data=np.ones((n,15))
    directions=[]
    grippers=[]
    with open (name,'rb') as csvfile:
        reader=csv.DictReader(csvfile)
        m=0
        for row in reader:
            data[m,0]=int(row['step'])
            data[m,1]=float(row['r_position_x'])
            data[m,2]=float(row['r_position_y'])
            data[m,3]=float(row['r_position_z'])
            data[m,4]=float(row['r_orientation_x'])
            data[m,5]=float(row['r_orientation_y'])
            data[m,6]=float(row['r_orientation_z'])
            data[m,7]=float(row['r_orientation_w'])
            data[m,8]=float(row['l_position_x'])
            data[m,9]=float(row['l_position_y'])
            data[m,10]=float(row['l_position_z'])
            data[m,11]=float(row['l_orientation_x'])
            data[m,12]=float(row['l_orientation_y'])
            data[m,13]=float(row['l_orientation_z'])
            data[m,14]=float(row['l_orientation_w'])
            directions.append(str(row['direction']))
            grippers.append(str(row['gripper']))
            m+=1
    print ('step len:',n)
    col_len=n
    n_epochs=1

    for epoch in range (n_epochs):
        for step in range (col_len):
            if grippers[step]=='r_o':
                rightgripper.open()
            if grippers[step]=='r_c':
                rightgripper.close()
            if grippers[step]=='l_o':
                leftgripper.open()
            if grippers[step]=='l_c':
                leftgripper.close()
            if grippers[step]=='rl_o':
                rightgripper.open()
                leftgripper.open()
            if grippers[step]=='l_c_r_o':
                leftgripper.close()
                rospy.sleep(2)
                rightgripper.open() 
            else:
                p.waitForSync()        
                pickgoal_r = PoseStamped()
                pickgoal_l=PoseStamped() 
                pickgoal_r.header.frame_id = "base"
                pickgoal_l.header.frame_id = "base"
                pickgoal_r.header.stamp = rospy.Time.now()
                pickgoal_l.header.stamp = rospy.Time.now()
                pickgoal_r.pose.position.x = data[step,1]
                pickgoal_r.pose.position.y = data[step,2]
                pickgoal_r.pose.position.z = data[step,3]
                pickgoal_r.pose.orientation.x = data[step,4]
                pickgoal_r.pose.orientation.y = data[step,5]
                pickgoal_r.pose.orientation.z = data[step,6]
                pickgoal_r.pose.orientation.w = data[step,7]
                pickgoal_l.pose.position.x = data[step,8]
                pickgoal_l.pose.position.y = data[step,9]
                pickgoal_l.pose.position.z = data[step,10]
                pickgoal_l.pose.orientation.x = data[step,11]
                pickgoal_l.pose.orientation.y = data[step,12]
                pickgoal_l.pose.orientation.z = data[step,13]
                pickgoal_l.pose.orientation.w = data[step,14]
                if directions[step]=='right':
                    gr.moveToPose(pickgoal_r, "right_gripper", plan_only=False)
                    rospy.sleep(2.0)
                if directions[step]=='left':
                    gl.moveToPose(pickgoal_l, "left_gripper",tolerance =0.05,plan_only=False)
                    rospy.sleep(2.0)
                if directions[step]=='both':
                    gr.moveToPose(pickgoal_r, "right_gripper", plan_only=False)
                    rospy.sleep(2.0)
                    gl.moveToPose(pickgoal_l, "left_gripper", plan_only=False)
                    rospy.sleep(2.0)
            if grippers[step]=='w_r_c':
                rightgripper.close()
            if grippers[step]=='w_l_c':
                leftgripper.close()
            if grippers[step]=='w_r_o_l_o':
                rightgripper.open()
                leftgripper.open()
            if grippers[step]=='w_r_o_l_c':
                leftgripper.close()
                rospy.sleep(2.0)
                rightgripper.open()
                
            print ('step',step+1,'finished, time:',time.time()-start_time)
            start_time=time.time()

class image_convert:
    def __init__(self,pos,num_Segmentation_towel):
        self.image_depth=message_filters.Subscriber("/camera/depth/image_raw",Image)
        self.image_rgb=message_filters.Subscriber("/camera/rgb/image_raw",Image)
        self.bridge=CvBridge()
        self.time_sychronization=message_filters.ApproximateTimeSynchronizer([self.image_depth,self.image_rgb],queue_size=10,slop=0.01,allow_headerless=True)
        self.start_time=time.time()
        self.pos=pos
        self.num_tw=num_Segmentation_towel

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
        image=cv_image_depth
        mask=np.ones(image.shape)*255
        for i in range(len(image)):
            for j in range(len(image[i])):
                    if 60<image[i][j]<65:
                        if 130<j<470 and i>80:
                            mask[i][j]=0
        rgb_mask=np.ones(image.shape)*255
        shift_step=10
        for i in range (len(rgb_mask)):
            for j in range (len(rgb_mask[i])-shift_step):
                rgb_mask[i][j]=mask[i][j+shift_step]
        cv_image_depth_real[mask>0]=0
        cv_image_rgb[rgb_mask>0]=0
        if time.time()-self.start_time>2:
            cv2.imwrite('/home/kentuen/Known_Configurations_datas/full_database/Segmentation/towel/pos_'+str(self.pos).zfill(4)+'/Segmentation/towel_'+str(self.num_tw).zfill(4)+'/'+str(time.time())+'_depth.png',cv_image_depth_real)
            cv2.imwrite('/home/kentuen/Known_Configurations_datas/full_database/Segmentation/towel/pos_'+str(self.pos).zfill(4)+'/Segmentation/towel_'+str(self.num_tw).zfill(4)+'/'+str(time.time())+'_rgb.png',cv_image_rgb)
            print ('Photo taken!')
            self.start_time=time.time()
        cv2.waitKey(3)
    
    def image_capture(self):
        print ('image capture starts...')
        self.time_sychronization.registerCallback(self.callback)

def pcl_cloud_point_callback(data):
    pc = ros_numpy.numpify(data)
    points=np.zeros((pc.shape[0],3))
    points[:,0]=pc['x']
    points[:,1]=pc['y']
    points[:,2]=pc['z']
    for i in range (len(points)):
        csv_writer.writerow((str(i+1),points[i,0],points[i,1],points[i,2]))

###########################################################################        
parser = argparse.ArgumentParser(description='Known Configurations Project')
parser.add_argument('--kcnet_model_no',type=int,default=100,help='kcnet model number')
parser.add_argument('--test_procceding',type=int,default=100,help='the number of the chosen test procceding')
parser.add_argument('--kc_shape',type=str,default='none',help='kcnet stage garment groud-truth shape')
parser.add_argument('--kc_pos',type=int,default=100,help='kcnet stage garment groud-truth pos')
args = parser.parse_args()

############################################################################
print ("Test begins")
print ("========================================================")
if os.path.exists('./cloud_points.csv'):
    sys.path.remove('./cloud_points.csv')
if args.test_procceding==100:
    print ("You must assign a test procceding, exiting...")
    exit()

if args.kc_shape=='none' or args.kc_pos==100:
    print ("You msut set a groud-truth shape and a groud-truth grasping point for your garment, exiting...")
    exit()

if args.kcnet_model_no==100:
    print ('You must assign a model number for kcnet, exiting...')
    exit()

if args.test_procceding==1:
##########GarNet Segmentation Stage##############
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
    print ('GarNet Segmentation Completes!')
    print ("========================================================")

############GarNet Stage########
    batch_size=32
    kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
    model=torch.load('./GarNet_Model/'+'model_01.pth')
    file_path='./GarNet_Database/depth'
    data='/'
    csv_path='./explore.csv'
    dataset=GarNet_Dataset(file_path+data,csv_path,transform=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((mean,),(std,))
    ]),opt=1)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=False,**kwargs)
    embeddings,labels, video_labels=extract_embeddings(dataloader,model)
    predicted_label,true_label=early_stop(embeddings,labels,video_labels,n=5)
    if not predicted_label==true_label:
        print ("GarNet Stage Fails, Exiting...")
        print ("========================================================")
        exit()
    print ("GarNet Stage Completes")
    print ("========================================================")
    print ("Test proceeding one completes, goes to robotic manipulation...")

if args.test_procceding==2:
########KCNet Data Capture######################
    kc_image_path='/home/kentuen/kcnet_garenet/original_images/'
    kc_list = os.listdir(kc_image_path)
    kc_image_name=kc_list[-1]
    image=cv2.imread(kc_image_name)
    target_path='/home/kentuen/kcnet_garenet/selected_images/'+args.kc_shape+'/pos_'+str(args.kc_pos).zfill(4)+'/image.png'
    cv2.imwrite(target_path,image)
    print ("KCNet segmentation completes!")
    print ("========================================================")

#############KCNet Stage#########################
    normalises=[0.02428423,0.02427759,0.02369768,0.02448228]
    stds=[0.0821249,0.08221505,0.08038522,0.0825848]
    category=args.kc_shape
    num_positions=1
    position_index=args.kc_pos-1
    num_frames=1
    shape=predicted_label

    acc=0
    correct=0
    for position in range(num_positions):
            for frame in range (num_frames):
                if category=='towel':
                    category_index=0
                elif category=='tshirt':
                    category_index=1
                elif category=='shirt':
                    category_index=2
                elif category=='sweater':
                    category_index=3
                elif category=='jean':
                    category_index=4
                else:
                    print ('category',category,'does not exit, exiting...')
                    break
                if num_positions==1:
                    images_add='./test_images/'+category+'/pos_'+str(position_index+1).zfill(4)+'/'+str(frame+1).zfill(4)+'.png'
                    true_label=category_index*10+position_index
                else:
                    images_add='./test_images/'+category+'/pos_'+str(position+1).zfill(4)+'/'+str(frame+1).zfill(4)+'.png'
                    true_label=category_index*10+position
                images=cv2.imread(images_add,0)
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((256,256)),
                    transforms.Normalize((normalises[args.model_no-1],), (stds[args.model_no-1],))
                ])
                data,shape=Get_Images(image=images,shape=shape,transforms=transform).__getitem__()
                shape=torch.from_numpy(shape).type(torch.float32)
                if num_positions==1:
                    pred,acc,correct=test(kcnet,data,shape,true_label,correct,acc,category,position_index)
                else:
                    pred,acc,correct=test(kcnet,data,shape,true_label,correct,acc,category,position)
    accuracy=100*(correct/acc)
    if num_positions !=1:
        print ('[category]',category,'[accuracy]',accuracy,'%')
    else:
        if accuracy==0:
            print ('Known Configuration Recognition Is  Failed, exiting...')
            print ("========================================================")
            exit()
        else:
            print ('Known Configuration Recognition Is Successful!')
    print ('KCNet stage completes!')
    print ("========================================================")

############Hand-Eye Calibration for Grasping Point for Left Hand###############
    target_path='./paths/'+CATEGORIES[pred.item()//10]+'/pos_'+str(int(pred.item()%10)).zfill(4)+'_stage_1.csv'
    pre_designed_manipulation=pd.read_csv(target_path).to_numpy()
    pre_designed_steps=0
    for idx in range(len(pre_designed_manipulation)):
        if pre_designed_manipulation[idx,16]=='w_r_o_l_c':
            pre_designed_point=pre_designed_manipulation[idx]
        pre_designed_steps+=1

    csv_data=pd.read_csv('cloud_points.csv').to_numpy()
    n_points=len(csv_data)
    csv_rev=np.flip(csv_data,axis=0)
    cloud_points_collection_x=[]
    cloud_points_collection_y=[]
    cloud_points_collection_z=[]
    state=None
    for idx in range(n_points):
        state=csv_data[idx,4]
        if state=='end':
            cloud_idx=idx+1
            while(state=='normal'):
                cloud_points_collection_x.append(csv_data[cloud_idx,1])
                cloud_points_collection_y.append(csv_data[cloud_idx,2])
                cloud_points_collection_z.append(csv_data[cloud_idx,3])
                state=csv_data[cloud_idx,4]
                cloud_idx+=1
            break
        if state=='end':
            break

    min=100
    for idx in range(len(cloud_points_collection_x)):
        dis=sqrt(pow((pre_designed_point[8]-cloud_points_collection_x[idx]),2)+pow((pre_designed_point[9]-cloud_points_collection_y[idx]),2)
        +pow((pre_designed_point[10]-cloud_points_collection_z[idx]),2))
        if dis<min:
            min=dis
            pre_designed_point[8]=cloud_points_collection_x[idx]
            pre_designed_point[9]=cloud_points_collection_y[idx]
            pre_designed_point[10]=cloud_points_collection_z[idx]

    if min=100:
        print ('Failed to find a grasping point (left), exiting...')
        print ("========================================================")
        exit()

    f=open('cabralited_manipulation_stage_1.csv','w')
    csv_writer=csv.writer(f)
    csv_writer.writerow('step','r_position_x','r_position_y','r_position_z','r_orientation_x','r_orientation_y', 'r_orientation_z',	'r_orientation_w','l_position_x','l_position_y','l_position_z'	
    ,'l_orientation_x','l_orientation_y','l_orientation_z','l_orientation_w','direction','gripper')

    for idx in range(len(pre_designed_manipulation)):
        if idx+1<pre_designed_steps:
        csv_writer(pre_designed_manipulation[idx,0],pre_designed_manipulation[idx,1],pre_designed_manipulation[idx,2],pre_designed_manipulation[idx,3],pre_designed_manipulation[idx,4],pre_designed_manipulation[idx,5],pre_designed_manipulation[idx,6],
        pre_designed_manipulation[idx,7],pre_designed_manipulation[idx,8],pre_designed_manipulation[idx,9],pre_designed_manipulation[idx,10],pre_designed_manipulation[idx,11],pre_designed_manipulation[idx,12],pre_designed_manipulation[idx,13],
        pre_designed_manipulation[idx,14],pre_designed_manipulation[idx,15],pre_designed_manipulation[idx,16])
        
    csv_writer(pre_designed_manipulation[pre_designed_steps,0],pre_designed_manipulation[pre_designed_steps,1],pre_designed_manipulation[pre_designed_steps,2],pre_designed_manipulation[pre_designed_steps,3],pre_designed_manipulation[pre_designed_steps,4],pre_designed_manipulation[pre_designed_steps,5],pre_designed_manipulation[pre_designed_steps,6],
        pre_designed_manipulation[pre_designed_steps,7],pre_designed_point[8],pre_designed_manipulation[pre_designed_steps,9],pre_designed_manipulation[pre_designed_steps,10],pre_designed_manipulation[pre_designed_steps,11],pre_designed_manipulation[pre_designed_steps,12],pre_designed_manipulation[pre_designed_steps,13],
        pre_designed_manipulation[pre_designed_steps,14],pre_designed_manipulation[pre_designed_steps,15],pre_designed_manipulation[pre_designed_steps,16])

    csv_writer(pre_designed_manipulation[pre_designed_steps,0],pre_designed_manipulation[pre_designed_steps,1],pre_designed_manipulation[pre_designed_steps,2],pre_designed_manipulation[pre_designed_steps,3],pre_designed_manipulation[pre_designed_steps,4],pre_designed_manipulation[pre_designed_steps,5],pre_designed_manipulation[pre_designed_steps,6],
        pre_designed_manipulation[pre_designed_steps,7],pre_designed_point[8],pre_designed_point[9],pre_designed_manipulation[pre_designed_steps,10],pre_designed_manipulation[pre_designed_steps,11],pre_designed_manipulation[pre_designed_steps,12],pre_designed_manipulation[pre_designed_steps,13],
        pre_designed_manipulation[pre_designed_steps,14],pre_designed_manipulation[pre_designed_steps,15],pre_designed_manipulation[pre_designed_steps,16])

    csv_writer(pre_designed_manipulation[pre_designed_steps,0],pre_designed_manipulation[pre_designed_steps,1],pre_designed_manipulation[pre_designed_steps,2],pre_designed_manipulation[pre_designed_steps,3],pre_designed_manipulation[pre_designed_steps,4],pre_designed_manipulation[pre_designed_steps,5],pre_designed_manipulation[pre_designed_steps,6],
        pre_designed_manipulation[pre_designed_steps,7],pre_designed_point[8],pre_designed_point[9],pre_designed_point[10],pre_designed_manipulation[pre_designed_steps,11],pre_designed_manipulation[pre_designed_steps,12],pre_designed_manipulation[pre_designed_steps,13],
        pre_designed_manipulation[pre_designed_steps,14],pre_designed_manipulation[pre_designed_steps,15],'w_l_o_r_c')
    print ("Left hand hand-eye calibrated!")
    print ("========================================================")
    print ("Test procceeding two completes, goes to robotic manipulation....")

if args.test_procceding==3:
###############Hand-Eye Calibration for Grasping Point for Right Hand#################
    target_path='./paths/'+CATEGORIES[pred.item()//10]+'/pos_'+str(int(pred.item()%10)).zfill(4)+'_stage_2.csv'
    pre_designed_manipulation=pd.read_csv(target_path).to_numpy()
    pre_designed_steps=0
    for idx in range(len(pre_designed_manipulation)):
        if pre_designed_manipulation[idx,16]=='w_r_c':
            pre_designed_point=pre_designed_manipulation[idx]
        pre_designed_steps+=1

    csv_data=pd.read_csv('cloud_points.csv').to_numpy()
    n_points=len(csv_data)
    csv_rev=np.flip(csv_data,axis=0)
    cloud_points_collection_x=[]
    cloud_points_collection_y=[]
    cloud_points_collection_z=[]
    state=None
    for idx in range(n_points):
        state=csv_data[idx,4]
        if state=='end':
            cloud_idx=idx+1
            while(state=='normal'):
                cloud_points_collection_x.append(csv_data[cloud_idx,1])
                cloud_points_collection_y.append(csv_data[cloud_idx,2])
                cloud_points_collection_z.append(csv_data[cloud_idx,3])
                state=csv_data[cloud_idx,4]
                cloud_idx+=1
            break
        if state=='end':
            break

    min=100
    for idx in range(len(cloud_points_collection_x)):
        dis=sqrt(pow((pre_designed_point[1]-cloud_points_collection_x[idx]),2)+pow((pre_designed_point[2]-cloud_points_collection_y[idx]),2)
        +pow((pre_designed_point[3]-cloud_points_collection_z[idx]),2))
        if dis<min:
            min=dis
            pre_designed_point[1]=cloud_points_collection_x[idx]
            pre_designed_point[2]=cloud_points_collection_y[idx]
            pre_designed_point[3]=cloud_points_collection_z[idx]

    if min=100:
        print ('Failed to find a grasping point (left), exiting...')
        print ("========================================================")
        exit()

    f=open('cabralited_manipulation_stage_2.csv','w')
    csv_writer=csv.writer(f)
    csv_writer.writerow('step','r_position_x','r_position_y','r_position_z','r_orientation_x','r_orientation_y', 'r_orientation_z',	'r_orientation_w','l_position_x','l_position_y','l_position_z'	
    ,'l_orientation_x','l_orientation_y','l_orientation_z','l_orientation_w','direction','gripper')

    for idx in range(len(pre_designed_manipulation)):
        if idx+1<pre_designed_steps:
        csv_writer(pre_designed_manipulation[idx,0],pre_designed_manipulation[idx,1],pre_designed_manipulation[idx,2],pre_designed_manipulation[idx,3],pre_designed_manipulation[idx,4],pre_designed_manipulation[idx,5],pre_designed_manipulation[idx,6],
        pre_designed_manipulation[idx,7],pre_designed_manipulation[idx,8],pre_designed_manipulation[idx,9],pre_designed_manipulation[idx,10],pre_designed_manipulation[idx,11],pre_designed_manipulation[idx,12],pre_designed_manipulation[idx,13],
        pre_designed_manipulation[idx,14],pre_designed_manipulation[idx,15],pre_designed_manipulation[idx,16])
        
    csv_writer(pre_designed_manipulation[idx,0],pre_designed_point[1],pre_designed_manipulation[idx,2],pre_designed_manipulation[idx,3],pre_designed_manipulation[idx,4],pre_designed_manipulation[idx,5],pre_designed_manipulation[idx,6],
        pre_designed_manipulation[idx,7],pre_designed_manipulation[idx,8],pre_designed_manipulation[idx,9],pre_designed_manipulation[idx,10],pre_designed_manipulation[idx,11],pre_designed_manipulation[idx,12],pre_designed_manipulation[idx,13],
        pre_designed_manipulation[idx,14],pre_designed_manipulation[idx,15],pre_designed_manipulation[idx,16])

    csv_writer(pre_designed_manipulation[idx,0],pre_designed_point[1],pre_designed_point[2],pre_designed_manipulation[idx,3],pre_designed_manipulation[idx,4],pre_designed_manipulation[idx,5],pre_designed_manipulation[idx,6],
        pre_designed_manipulation[idx,7],pre_designed_manipulation[idx,8],pre_designed_manipulation[idx,9],pre_designed_manipulation[idx,10],pre_designed_manipulation[idx,11],pre_designed_manipulation[idx,12],pre_designed_manipulation[idx,13],
        pre_designed_manipulation[idx,14],pre_designed_manipulation[idx,15],pre_designed_manipulation[idx,16])

    csv_writer(pre_designed_manipulation[idx,0],pre_designed_point[1],pre_designed_point[2],pre_designed_point[3],pre_designed_manipulation[idx,4],pre_designed_manipulation[idx,5],pre_designed_manipulation[idx,6],
        pre_designed_manipulation[idx,7],pre_designed_manipulation[idx,8],pre_designed_manipulation[idx,9],pre_designed_manipulation[idx,10],pre_designed_manipulation[idx,11],pre_designed_manipulation[idx,12],pre_designed_manipulation[idx,13],
        pre_designed_manipulation[idx,14],pre_designed_manipulation[idx,15],'w_r_c')
    print ("Right hand hand-eye calibrated!")
    print ("========================================================")
    print ("Test proceeding three completes, goes to robotic manipulation...")
###################################################



