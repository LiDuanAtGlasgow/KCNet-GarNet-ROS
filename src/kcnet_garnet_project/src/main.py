#type:ignore
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
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
from continuous_perception import early_stop
import csv
import sys
from sklearn.preprocessing import OneHotEncoder
import argparse
import glob
import torchvision.models as models
import torch.nn.functional as F
import math

np.random.seed(42)
torch.manual_seed(42)
cuda=False

def extract_embeddings_from_csv(csv_file,dataloader,model):
    embeddings_seen=csv_file[:,1:3]
    labels_seen=csv_file[:,3]
    video_labels_seen=csv_file[:,4]
    with torch.no_grad():
        model.eval()
        embeddings=np.zeros((len(dataloader.dataset),2))
        labels=np.zeros(len(dataloader.dataset))
        video_labels=np.zeros(len(dataloader.dataset))
        k=0
        for images, target, video_label in dataloader:
            embeddings[k:k+len(images)]=model.get_emdding(images).data.cpu().numpy()
            labels[k:k+len(images)]=target.numpy()
            video_labels[k:k+len(images)]=video_label.numpy()
            k+=len(images)
    embeddings=np.concatenate((embeddings_seen,embeddings),axis=0)
    labels=np.concatenate((labels_seen,labels),axis=0)
    video_labels=np.concatenate((video_labels_seen,video_labels),axis=0)
    return embeddings,labels,video_labels

class ResNet18_Embedding(nn.Module):
    def __init__(self):
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
        img=img.cpu().detach().numpy()
        video_label=int(self.video_labels[index])
        return img, target, video_label
    
    def __len__(self):
        return len(self.data)

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
    plt.show()

enc=OneHotEncoder(handle_unknown='ignore')
categories=[['jean'],['shirt'],['sweater'],['tshirt'],['towel']]
enc.fit(categories)

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

class KCNet(nn.Module):
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

def test(kcnet,data,shape,true_label,category,position_index):
    output=kcnet(data,shape)
    pred=output.argmax(dim=1,keepdim=True)
    print ('true_postion:',category,position_index+1)
    print('predicted_postion:',CATEGORIES[pred.item()//10],pred.item()%10+1)
    if true_label==pred.item():
        correct=True
    else:
        correct=False
    return pred,correct

###########################################################################        
parser = argparse.ArgumentParser(description='Known Configurations Project')
parser.add_argument('--kcnet_model_no',type=int,default=100,help='kcnet model number')
parser.add_argument('--test_procceding',type=int,default=100,help='the number of the chosen test procceding')
parser.add_argument('--kc_shape',type=str,default='none',help='kcnet stage garment groud-truth shape')
parser.add_argument('--kc_pos',type=int,default=100,help='kcnet stage garment groud-truth pos')
parser.add_argument('--garnet_model_no',type=int,default=100,help='garnet model number')
parser.add_argument('--garnet_shape',type=int,default=100,help='garnet stage garment groud-truth shape')
parser.add_argument('--garnet_video_idx',type=int,default=11,help='garnet stage unseen garment index')
args = parser.parse_args()

############################################################################
print ("========================================================")
print ("Test begins")
print ("========================================================")
if args.test_procceding==100:
    print ("You must assign a test procceding, exiting...")
    exit()
if args.kc_shape=='none' or args.kc_pos==100:
    print ("You msut set a groud-truth shape and a groud-truth grasping point for your garment, exiting...")
    exit()
if args.kcnet_model_no==100:
    print ('You must assign a model number for kcnet, exiting...')
    exit()
if args.garnet_model_no==100:
    print ('You must assign a model number for garnet, exiting...')
    exit()
if args.garnet_shape==100:
    print ('You must assign a shape number for garnet, exiting...')
    exit()

device = torch.device('cpu')
garnet_predicted_shape=None

if args.test_procceding==1:
##########GarNet Segmentation Stage##############
    shapes=['pant','shirt','sweater','towel','tshirt']
    shape_label=shapes[args.garnet_shape]
    f=open('./garnet_explore_file/no_'+str(args.garnet_shape+1).zfill(2)+'/explore.csv','w')
    csv_writer=csv.writer(f)
    csv_writer.writerow(('name','shape','discretised_weight','video_idx'))
    depth_folder='./depth_raw_images/'
    masked_depth_folder='./garnet_database/'
    if not os.path.exists(depth_folder):
        os.makedirs(depth_folder)
    if not os.path.exists(masked_depth_folder):
        os.makedirs(masked_depth_folder)

    i=0
    for fil_name in sorted(glob.glob('/home/kentuen/ros_ws/src/kcnet_garnet_project/src/raw_images/*_depth.png'),key=str.lower):
        image=cv2.imread(fil_name)
        image_path=depth_folder+shape_label+'_garnet_kcnet_test_'+str(i+1).zfill(4)+'.png'
        cv2.imwrite(image_path,image)
        csv_writer.writerow((shape_label+'_garnet_kcnet_test_'+str(i+1).zfill(4)+'.png',11+args.garnet_shape,0,args.garnet_video_idx))
        i+=1

    start_time=time.time()
    num_images=len(next(os.walk(depth_folder))[2])
    for idx in range(num_images):
        image_path=depth_folder+shape_label+'_garnet_kcnet_test_'+str(idx+1).zfill(4)+'.png'
        image=cv2.imread(image_path,0)
        mask=np.ones(image.shape)*255
        for i in range(len(image)):
            if idx<=15:
                for j in range(len(image[i])):
                    if 0<image[i][j]<55:
                        if 100<j<360 and 300>i>270:
                            mask[i][j]=0
            if 15<idx:
                for j in range(len(image[i])):
                    if 0<image[i][j]<55:
                        if 100<j<360 and 300>i>270-int(((250/45)*(idx-14))):
                            mask[i][j]=0

        depth_image=cv2.imread(depth_folder+shape_label+'_garnet_kcnet_test_'+str(idx+1).zfill(4)+'.png')
        depth_image[mask>0]=0
        cv2.imwrite(masked_depth_folder+shape_label+'_garnet_kcnet_test_'+str(idx+1).zfill(4)+'.png',depth_image)
        if idx%int(num_images/10)==0:
            print('No',idx+1,'has been finished! time consumed:',time.time()-start_time)
            start_time=time.time()
    f.close()
    print ("========================================================")
    print ('Garnet segmentation completed!')
    print ("========================================================")

############GarNet Stage########
    garnet_mean,garnet_std=0.00586554,0.03234654
    batch_size=32
    confid_circs=[[-30,50,-30,30,60,80],[-30,30,-30,40,70,60],[-20,30,-20,30,50,50],[-40,30,-50,20,70,70]]
    kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
    model=torch.load('/home/kentuen/ros_ws/src/kcnet_garnet_project/src/garnet_model/'+'no_'+str(args.garnet_model_no).zfill(2)+'.pth',map_location=device)
    file_path='/home/kentuen/ros_ws/src/kcnet_garnet_project/src/garnet_database/'
    csv_path='/home/kentuen/ros_ws/src/kcnet_garnet_project/src/garnet_explore_file/no_'+str(args.garnet_model_no).zfill(2)+'/explore.csv'
    dataset=GarNet_Dataset(file_path,csv_path,transform=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((garnet_mean,),(garnet_std,))
    ]),opt=1)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=False,**kwargs)
    csv_file=pd.read_csv('./garnet_embeddings/embeddings_no_'+str(args.garnet_model_no).zfill(2)+'.csv').to_numpy()
    embeddings,labels, video_labels=extract_embeddings_from_csv(csv_file,dataloader,model)
    confid_circ=confid_circs[args.garnet_model_no-1]
    predicted_label,true_label=early_stop(embeddings,labels,video_labels,confid_circ=confid_circ,
    category_idx=args.garnet_shape,video_idx=args.garnet_video_idx)
    if not predicted_label==true_label:
        print ("Garnet stage failed, exiting...")
        print ("========================================================")
        exit()
    garnet_predicted_shape=predicted_label
    f=open('garnet_predicted_shapes.csv','w')
    csv_writer=csv.writer(f)
    csv_writer.writerow(('no','shape'))
    csv_writer.writerow((1,predicted_label))
    print ("Garnet stage completed")
    print ("========================================================")
    print ("Test proceeding one completed, go to robotic manipulation (route_GarNet2KCNet)...")
    print ("========================================================")

if args.test_procceding==2:
########KCNet Data Capture######################
    kc_image_path='/home/kentuen/ros_ws/src/kcnet_garnet_project/src/kcnet_original_images/'
    kc_list = os.listdir(kc_image_path)
    kc_image_name=kc_image_path+kc_list[-1]
    image=cv2.imread(kc_image_name)
    if os.path.exists('/home/kentuen/ros_ws/src/kcnet_garnet_project/src/kcnet_selected_images/'+args.kc_shape+'/pos_'+str(args.kc_pos).zfill(4)+'/image.png'):
        os.remove('/home/kentuen/ros_ws/src/kcnet_garnet_project/src/kcnet_selected_images/'+args.kc_shape+'/pos_'+str(args.kc_pos).zfill(4)+'/image.png')
    if not os.path.exists('/home/kentuen/ros_ws/src/kcnet_garnet_project/src/kcnet_selected_images/'+args.kc_shape+'/pos_'+str(args.kc_pos).zfill(4)+'/'):
        os.makedirs('/home/kentuen/ros_ws/src/kcnet_garnet_project/src/kcnet_selected_images/'+args.kc_shape+'/pos_'+str(args.kc_pos).zfill(4)+'/')
    target_path='/home/kentuen/ros_ws/src/kcnet_garnet_project/src/kcnet_selected_images/'+args.kc_shape+'/pos_'+str(args.kc_pos).zfill(4)+'/image.png'
    cv2.imwrite(target_path,image)
    print ("Kcnet segmentation completed!")
    print ("========================================================")

#############KCNet Stage#########################
    normalises=[0.02428423,0.02427759,0.02369768,0.02448228]
    stds=[0.0821249,0.08221505,0.08038522,0.0825848]
    category=args.kc_shape
    position_index=args.kc_pos-1
    shape=pd.read_csv('garnet_predicted_shapes.csv').to_numpy()[0][1]
    print('shape:',shape)

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
        exit()
    
    kcnet=KCNet()
    kcnet.load_state_dict(torch.load('./kcnet_model/no_'+str(args.kcnet_model_no)+'.pt',map_location=device))
    kcnet.eval()
    images_add=target_path
    true_label=category_index*10+position_index
    images=cv2.imread(images_add,0)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256)),
        transforms.Normalize((normalises[args.kcnet_model_no-1],), (stds[args.kcnet_model_no-1],))
    ])
    data,shape=Get_Images(image=images,shape=shape,transforms=transform).__getitem__()
    shape=torch.from_numpy(shape).type(torch.float32)
    pred,correct=test(kcnet,data,shape,true_label,category,position_index=position_index)
    if not correct:
        print ('Known configuration recognition is failed, exiting...')
        print ("========================================================")
        exit()
    
    f=open('kcnet_predicted_kc.csv','w')
    csv_writer=csv.writer(f)
    csv_writer.writerow('no','kc_no')
    csv_writer.writerow(1,pred.item())
    print ('known configuration recognition is successful!')    
    print ('Kcnet stage completed!')
    print ("========================================================")
    print ("Test proceeding two completed, go to robotic manipulation (calibrated_manipulation_stage_1)...")
    print ("========================================================")

############Hand-Eye Calibration for Grasping Point for Left Hand###############
if args.test_procceding==3:
    pred=pd.read_csv('kcnet_predicted_kc.csv').to_numpy()[0][1]
    target_path='./paths/'+CATEGORIES[pred//10]+'/pos_'+str(int(pred%10+1)).zfill(4)+'/stage_2.csv'
    pre_designed_manipulation=pd.read_csv(target_path).to_numpy()
    pre_designed_steps=0
    pre_designed_key_step=0
    for idx in range(len(pre_designed_manipulation)):
        if pre_designed_manipulation[idx,16]=='w_r_o_l_c':
            pre_designed_point=pre_designed_manipulation[idx]
            pre_designed_key_step=idx
        pre_designed_steps+=1

    csv_data=pd.read_csv('./cloud_points.csv').to_numpy()
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
            state=csv_data[idx+1,4]
            i=0
            while(state=='normal'):
                cloud_points_collection_x.append(csv_data[cloud_idx,1])
                cloud_points_collection_y.append(csv_data[cloud_idx,2])
                cloud_points_collection_z.append(csv_data[cloud_idx,3])
                state=csv_data[cloud_idx+1,4]
                cloud_idx+=1
                i+=1
        if state=='end':
            break
    
    min=100
    chosen_idx=0
    pre_designed_point_=np.zeros(pre_designed_point.shape)
    for idx in range(len(cloud_points_collection_x)):
        dis=math.sqrt(pow((float(pre_designed_point[8])-float(cloud_points_collection_x[idx])),2)+pow((float(pre_designed_point[9])-float(cloud_points_collection_y[idx])),2)
        +pow((float(pre_designed_point[10])-float(cloud_points_collection_z[idx])),2))        
        if dis<min:
            min=dis
            chosen_idx=idx
            pre_designed_point_[8]=float(cloud_points_collection_x[idx])
            pre_designed_point_[9]=float(cloud_points_collection_y[idx])
            pre_designed_point_[10]=float(cloud_points_collection_z[idx])
    
    print ("pre_designed_point[8]",float(pre_designed_point[8]),", cloud_points_collection_x[chosen_idx]",float(cloud_points_collection_x[chosen_idx]))
    print ("pre_designed_point[9]",float(pre_designed_point[9]),", cloud_points_collection_y[chosen_idx]",float(cloud_points_collection_y[chosen_idx]))
    print ("pre_designed_point[10]",float(pre_designed_point[10]),", cloud_points_collection_z[chosen_idx]",float(cloud_points_collection_z[chosen_idx]))
    print ("========================================================")
    if abs(float(pre_designed_point[8])-float(cloud_points_collection_x[chosen_idx]))>0.3 or abs(float(pre_designed_point[9])-float(cloud_points_collection_y[chosen_idx]))>0.3 or abs(float(pre_designed_point[10])-float(cloud_points_collection_z[chosen_idx]))>0.3:
        print("Disances are too large, failed, exiting...")
        print ("========================================================")
        exit()
    
    if min==100:
        print ('Failed to find a grasping point (left), exiting...')
        print ("========================================================")
        exit()  

    f=open('/home/kentuen/ros_ws/src/robot_manipulation/scripts/manipulation/calibrated_manipulation_stage_1.csv','w')
    csv_writer=csv.writer(f)
    csv_writer.writerow(('step','r_position_x','r_position_y','r_position_z','r_orientation_x','r_orientation_y', 'r_orientation_z',	'r_orientation_w','l_position_x','l_position_y','l_position_z'	
    ,'l_orientation_x','l_orientation_y','l_orientation_z','l_orientation_w','direction','gripper'))

    for idx in range(len(pre_designed_manipulation)):
        if idx+1<=pre_designed_key_step:
            csv_writer.writerow((pre_designed_manipulation[idx,0],pre_designed_manipulation[idx,1],pre_designed_manipulation[idx,2],pre_designed_manipulation[idx,3],pre_designed_manipulation[idx,4],pre_designed_manipulation[idx,5],pre_designed_manipulation[idx,6],
            pre_designed_manipulation[idx,7],pre_designed_manipulation[idx,8],pre_designed_manipulation[idx,9],pre_designed_manipulation[idx,10],pre_designed_manipulation[idx,11],pre_designed_manipulation[idx,12],pre_designed_manipulation[idx,13],
            pre_designed_manipulation[idx,14],pre_designed_manipulation[idx,15],pre_designed_manipulation[idx,16]))
        
    csv_writer.writerow((pre_designed_manipulation[pre_designed_key_step,0],pre_designed_manipulation[pre_designed_key_step,1],pre_designed_manipulation[pre_designed_key_step,2],pre_designed_manipulation[pre_designed_key_step,3],pre_designed_manipulation[pre_designed_key_step,4],pre_designed_manipulation[pre_designed_key_step,5],pre_designed_manipulation[pre_designed_key_step,6],
        pre_designed_manipulation[pre_designed_key_step,7],pre_designed_point_[8],pre_designed_manipulation[pre_designed_key_step,9],pre_designed_manipulation[pre_designed_key_step,10],pre_designed_manipulation[pre_designed_key_step,11],pre_designed_manipulation[pre_designed_key_step,12],pre_designed_manipulation[pre_designed_key_step,13],
        pre_designed_manipulation[pre_designed_key_step,14],pre_designed_manipulation[pre_designed_key_step,15],'open'))

    csv_writer.writerow((pre_designed_manipulation[pre_designed_key_step,0],pre_designed_manipulation[pre_designed_key_step,1],pre_designed_manipulation[pre_designed_key_step,2],pre_designed_manipulation[pre_designed_key_step,3],pre_designed_manipulation[pre_designed_key_step,4],pre_designed_manipulation[pre_designed_key_step,5],pre_designed_manipulation[pre_designed_key_step,6],
        pre_designed_manipulation[pre_designed_key_step,7],pre_designed_point_[8],pre_designed_point_[9],pre_designed_manipulation[pre_designed_key_step,10],pre_designed_manipulation[pre_designed_key_step,11],pre_designed_manipulation[pre_designed_key_step,12],pre_designed_manipulation[pre_designed_key_step,13],
        pre_designed_manipulation[pre_designed_key_step,14],pre_designed_manipulation[pre_designed_key_step,15],'open'))

    csv_writer.writerow((pre_designed_manipulation[pre_designed_key_step,0],pre_designed_manipulation[pre_designed_key_step,1],pre_designed_manipulation[pre_designed_key_step,2],pre_designed_manipulation[pre_designed_key_step,3],pre_designed_manipulation[pre_designed_key_step,4],pre_designed_manipulation[pre_designed_key_step,5],pre_designed_manipulation[pre_designed_key_step,6],
        pre_designed_manipulation[pre_designed_key_step,7],pre_designed_point_[8],pre_designed_point_[9],pre_designed_point_[10],pre_designed_manipulation[pre_designed_key_step,11],pre_designed_manipulation[pre_designed_key_step,12],pre_designed_manipulation[pre_designed_key_step,13],
        pre_designed_manipulation[pre_designed_key_step,14],pre_designed_manipulation[pre_designed_key_step,15],'w_r_o_l_c'))
    
    for idx in range(len(pre_designed_manipulation)):
        if idx>pre_designed_key_step:
            csv_writer.writerow((pre_designed_manipulation[idx,0],pre_designed_manipulation[idx,1],pre_designed_manipulation[idx,2],pre_designed_manipulation[idx,3],pre_designed_manipulation[idx,4],pre_designed_manipulation[idx,5],pre_designed_manipulation[idx,6],
            pre_designed_manipulation[idx,7],pre_designed_manipulation[idx,8],pre_designed_manipulation[idx,9],pre_designed_manipulation[idx,10],pre_designed_manipulation[idx,11],pre_designed_manipulation[idx,12],pre_designed_manipulation[idx,13],
            pre_designed_manipulation[idx,14],pre_designed_manipulation[idx,15],pre_designed_manipulation[idx,16]))

    print ("The left gripper hand-eye calibrated!")
    print ("========================================================")
    print ("Test procceeding three completed, go to robotic manipulation (calibrated_manipulation_stage_2)....")
    print ("========================================================")

if args.test_procceding==4:
###############Hand-Eye Calibration for Grasping Point for Right Hand#################
    pred=pd.read_csv('kcnet_predicted_kc.csv').to_numpy()[0][1]
    target_path='./paths/'+CATEGORIES[pred//10]+'/pos_'+str(int(pred%10+1)).zfill(4)+'/stage_3.csv'
    pre_designed_manipulation=pd.read_csv(target_path).to_numpy()
    pre_designed_steps=0
    pre_designed_key_step=0
    for idx in range(len(pre_designed_manipulation)):
        if pre_designed_manipulation[idx,16]=='w_r_c':
            pre_designed_point=pre_designed_manipulation[idx]
            pre_designed_key_step=idx
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
            state=csv_data[idx+1,4]
            while(state=='normal'):
                cloud_points_collection_x.append(csv_data[cloud_idx,1])
                cloud_points_collection_y.append(csv_data[cloud_idx,2])
                cloud_points_collection_z.append(csv_data[cloud_idx,3])
                state=csv_data[cloud_idx+1,4]
                cloud_idx+=1
            break
        if state=='end':
            break

    min=100
    pre_designed_point_=np.zeros(pre_designed_point.shape)
    chosen_idx=0
    for idx in range(len(cloud_points_collection_x)):
        dis=math.sqrt(pow((float(pre_designed_point[1])-float(cloud_points_collection_x[idx])),2)+pow((float(pre_designed_point[2])-float(cloud_points_collection_y[idx])),2)
        +pow((float(pre_designed_point[3])-float(cloud_points_collection_z[idx])),2))
        if dis<min:
            min=dis
            pre_designed_point_[1]=cloud_points_collection_x[idx]
            pre_designed_point_[2]=cloud_points_collection_y[idx]
            pre_designed_point_[3]=cloud_points_collection_z[idx]
            chosen_idx=idx

    print ("pre_designed_point[1]",float(pre_designed_point[1]),", cloud_points_collection_x[chosen_idx]",float(cloud_points_collection_x[chosen_idx]))
    print ("pre_designed_point[2]",float(pre_designed_point[2]),", cloud_points_collection_y[chosen_idx]",float(cloud_points_collection_y[chosen_idx]))
    print ("pre_designed_point[3]",float(pre_designed_point[3]),", cloud_points_collection_z[chosen_idx]",float(cloud_points_collection_z[chosen_idx]))
    print ("========================================================")
    if abs(float(pre_designed_point[1])-float(cloud_points_collection_x[chosen_idx]))>0.3 or abs(float(pre_designed_point[2])-float(cloud_points_collection_y[chosen_idx]))>0.3 or abs(float(pre_designed_point[3])-float(cloud_points_collection_z[chosen_idx]))>0.3:
        print("disances are too large, failed, exiting...")
        print ("========================================================")
        exit()


    if min==100:
        print ('Failed to find a grasping point (left), exiting...')
        print ("========================================================")
        exit()

    f=open('/home/kentuen/ros_ws/src/robot_manipulation/scripts/manipulation/calibrated_manipulation_stage_2.csv','w')
    csv_writer=csv.writer(f)
    csv_writer.writerow(('step','r_position_x','r_position_y','r_position_z','r_orientation_x','r_orientation_y', 'r_orientation_z',	'r_orientation_w','l_position_x','l_position_y','l_position_z'	
    ,'l_orientation_x','l_orientation_y','l_orientation_z','l_orientation_w','direction','gripper'))

    for idx in range(len(pre_designed_manipulation)):
        if idx+1<=pre_designed_key_step:
            csv_writer.writerow((pre_designed_manipulation[idx,0],pre_designed_manipulation[idx,1],pre_designed_manipulation[idx,2],pre_designed_manipulation[idx,3],pre_designed_manipulation[idx,4],pre_designed_manipulation[idx,5],pre_designed_manipulation[idx,6],
            pre_designed_manipulation[idx,7],pre_designed_manipulation[idx,8],pre_designed_manipulation[idx,9],pre_designed_manipulation[idx,10],pre_designed_manipulation[idx,11],pre_designed_manipulation[idx,12],pre_designed_manipulation[idx,13],
            pre_designed_manipulation[idx,14],pre_designed_manipulation[idx,15],pre_designed_manipulation[idx,16]))    
    
    csv_writer.writerow((pre_designed_manipulation[pre_designed_key_step,0],pre_designed_point_[1],pre_designed_manipulation[pre_designed_key_step,2],pre_designed_manipulation[pre_designed_key_step,3],pre_designed_manipulation[pre_designed_key_step,4],pre_designed_manipulation[pre_designed_key_step,5],pre_designed_manipulation[pre_designed_key_step,6],
        pre_designed_manipulation[pre_designed_key_step,7],pre_designed_manipulation[pre_designed_key_step,8],pre_designed_manipulation[pre_designed_key_step,9],pre_designed_manipulation[pre_designed_key_step,10],pre_designed_manipulation[pre_designed_key_step,11],pre_designed_manipulation[pre_designed_key_step,12],pre_designed_manipulation[pre_designed_key_step,13],
        pre_designed_manipulation[pre_designed_key_step,14],pre_designed_manipulation[pre_designed_key_step,15],'open'))
    
    csv_writer.writerow((pre_designed_manipulation[pre_designed_key_step,0],pre_designed_point_[1],pre_designed_point_[2],pre_designed_manipulation[pre_designed_key_step,3],pre_designed_manipulation[pre_designed_key_step,4],pre_designed_manipulation[pre_designed_key_step,5],pre_designed_manipulation[pre_designed_key_step,6],
        pre_designed_manipulation[pre_designed_key_step,7],pre_designed_manipulation[pre_designed_key_step,8],pre_designed_manipulation[pre_designed_key_step,9],pre_designed_manipulation[pre_designed_key_step,10],pre_designed_manipulation[pre_designed_key_step,11],pre_designed_manipulation[pre_designed_key_step,12],pre_designed_manipulation[pre_designed_key_step,13],
        pre_designed_manipulation[pre_designed_key_step,14],pre_designed_manipulation[pre_designed_key_step,15],'open'))
    
    csv_writer.writerow((pre_designed_manipulation[pre_designed_key_step,0],pre_designed_point_[1],pre_designed_point_[2],pre_designed_point_[3],pre_designed_manipulation[pre_designed_key_step,4],pre_designed_manipulation[pre_designed_key_step,5],pre_designed_manipulation[pre_designed_key_step,6],
        pre_designed_manipulation[pre_designed_key_step,7],pre_designed_manipulation[pre_designed_key_step,8],pre_designed_manipulation[pre_designed_key_step,9],pre_designed_manipulation[pre_designed_key_step,10],pre_designed_manipulation[pre_designed_key_step,11],pre_designed_manipulation[pre_designed_key_step,12],pre_designed_manipulation[pre_designed_key_step,13],
        pre_designed_manipulation[pre_designed_key_step,14],pre_designed_manipulation[pre_designed_key_step,15],'w_r_c')) 
    for idx in range(len(pre_designed_manipulation)):
        if idx>pre_designed_key_step:
            csv_writer.writerow((pre_designed_manipulation[idx,0],pre_designed_manipulation[idx,1],pre_designed_manipulation[idx,2],pre_designed_manipulation[idx,3],pre_designed_manipulation[idx,4],pre_designed_manipulation[idx,5],pre_designed_manipulation[idx,6],
            pre_designed_manipulation[idx,7],pre_designed_manipulation[idx,8],pre_designed_manipulation[idx,9],pre_designed_manipulation[idx,10],pre_designed_manipulation[idx,11],pre_designed_manipulation[idx,12],pre_designed_manipulation[idx,13],
            pre_designed_manipulation[idx,14],pre_designed_manipulation[idx,15],pre_designed_manipulation[idx,16]))
    print ("The right gripper hand-eye calibrated!")
    print ("========================================================")
    print ("Test proceeding four completed, go to robotic manipulation (calibrated_manipulation_stage_3)...")
    print ("========================================================")
###################################################



