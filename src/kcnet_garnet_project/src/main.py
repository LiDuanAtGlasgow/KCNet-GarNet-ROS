#type:ignore
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

np.random.seed(42)
torch.manual_seed(42)
cuda=torch.cuda.is_available()

def extract_embeddings_from_csv(csv_file,dataloader,model):
    embeddings_seen=csv_file[:,1:3]
    labels_seen=labels[:,3]
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

def test(kcnet,data,true_label,category,position_index):
    output=kcnet(data)
    pred=output.argmax(dim=1,keepdim=True)
    print ('true_postion:',category,position_index)
    print('predicted_postion:',CATEGORIES[pred.item()//10],pred.item()%10)
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
if args.garnet_model_no==100:
    print ('You must assign a model number for garnet, exiting...')
    exit()
if args.garnet_shape==100:
    print ('You must assign a shape number for garnet, exiting...')
    exit()

device = torch.device('cpu')

if args.test_procceding==1:
##########GarNet Segmentation Stage##############
    shapes=['pant','shirt','sweater','towel','tshirt']
    shape_label=shapes[args.garent_shape]
    f=csv.open('./garnet_explore_file/no_'+str(args.garnet_shape+1).zfill(3)+'/explore.csv','a')
    csv_writer=csv.writer(f)
    depth_folder='./depth_images/'
    rgb_folder='./rgb_images/'
    masked_depth_folder='./garnet_database/depth/'
    masked_rgb_folder='./garnet_database/rgb/'
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
        image_path=depth_folder+shape_label+'garnet_kcnet_test_'+str(i+1).zfill(4)+'.png'
        i+=1
        cv2.imwrite(image_path,image)
        csv_writer.writerows(shape_label+'garnet_kcnet_test_'+str(i+1).zfill(4)+'.png',11+args.garnet_shape,0,args.garnet_video_idx)

    i=0
    for fil_name in sorted(glob.glob('./images/*_rgb.png'),key=str.lower):
        image=cv2.imread(fil_name)
        image_path=rgb_folder+shape_label+'garnet_kcnet_test_'+str(i+1).zfill(4)+'.png'
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
    garnet_mean,garnet_std=0.00586554,0.03234654
    batch_size=32
    confid_circs=[[-30,50,-30,30,60,80],[-30,30,-30,40,70,60],[-20,30,-20,30,50,50],[-40,30,-50,20,70,70]]
    kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
    model=torch.load('./garnet_model/'+'model_'+str(args.garnet_model_no).zfill(2)+'.pth',map_location=device)
    file_path='./garnet_database/'
    csv_path='./garnet_explore_file/no_'+str(args.garnet_shape+1).zfill(2)+'/explore.csv'
    dataset=GarNet_Dataset(file_path,csv_path,transform=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((garnet_mean,),(garnet_std,))
    ]),opt=1)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=False,**kwargs)
    csv_file=pd.read_csv('./garnet_embeddings/embeddings_no_'+str(args.garnet_shape+1).zfill(2)+'.csv')
    embeddings,labels, video_labels=extract_embeddings_from_csv(csv_file,dataloader,model)
    confid_circ=confid_circs[args.garent_model_no]
    predicted_label,true_label=early_stop(embeddings,labels,video_labels,confid_circ=confid_circ,
    category_idx=args.garnet_shape,video_idx=args.garnet_video_idx)
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
    position_index=args.kc_pos-1
    shape=predicted_label

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
    
    kcnet=KCNet()
    kcnet.load_state_dict(torch.load('./kcnet_model/no_'+str(args.model_no)+'.pt',map_location=device))
    kcnet.eval()
    images_add='./kcnet_test_images/'+args.kc_shape+'/pos_'+str(args.kc_pos).zfill(4)+'/image.png'
    true_label=category_index*10+position_index
    images=cv2.imread(images_add,0)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256)),
        transforms.Normalize((normalises[args.model_no-1],), (stds[args.model_no-1],))
    ])
    data,shape=Get_Images(image=images,shape=shape,transforms=transform).__getitem__()
    shape=torch.from_numpy(shape).type(torch.float32)
    pred,correct=test(kcnet,data,shape,true_label,category,position_index)
    if not correct:
        print ('Known Configuration Recognition Is  Failed, exiting...')
        print ("========================================================")
        exit()
    else:
        print ('Known Configuration Recognition Is Successful!')
    print ('KCNet stage completes!')
    print ("========================================================")

############Hand-Eye Calibration for Grasping Point for Left Hand###############
    target_path='./paths/'+CATEGORIES[pred.item()//10]+'/pos_'+str(int(pred.item()%10)).zfill(4)+'/stage_1.csv'
    pre_designed_manipulation=pd.read_csv(target_path).to_numpy()
    pre_designed_steps=0
    pre_designed_key_step=0
    for idx in range(len(pre_designed_manipulation)):
        if pre_designed_manipulation[idx,16]=='w_r_o_l_c':
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
        if idx+1<=pre_designed_key_step:
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
    
    for idx in range(len(pre_designed_manipulation)):
        if idx+1>pre_designed_key_step:
            csv_writer(pre_designed_manipulation[idx,0],pre_designed_manipulation[idx,1],pre_designed_manipulation[idx,2],pre_designed_manipulation[idx,3],pre_designed_manipulation[idx,4],pre_designed_manipulation[idx,5],pre_designed_manipulation[idx,6],
            pre_designed_manipulation[idx,7],pre_designed_manipulation[idx,8],pre_designed_manipulation[idx,9],pre_designed_manipulation[idx,10],pre_designed_manipulation[idx,11],pre_designed_manipulation[idx,12],pre_designed_manipulation[idx,13],
            pre_designed_manipulation[idx,14],pre_designed_manipulation[idx,15],pre_designed_manipulation[idx,16])

    print ("Left hand hand-eye calibrated!")
    print ("========================================================")
    print ("Test procceeding two completes, goes to robotic manipulation....")

if args.test_procceding==3:
###############Hand-Eye Calibration for Grasping Point for Right Hand#################
    target_path='./paths/'+CATEGORIES[pred.item()//10]+'/pos_'+str(int(pred.item()%10)).zfill(4)+'/stage_2.csv'
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
        if idx+1<=pre_designed_key_step:
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
    for idx in range(len(pre_designed_manipulation)):
        if idx+1>pre_designed_key_step:
            csv_writer(pre_designed_manipulation[idx,0],pre_designed_manipulation[idx,1],pre_designed_manipulation[idx,2],pre_designed_manipulation[idx,3],pre_designed_manipulation[idx,4],pre_designed_manipulation[idx,5],pre_designed_manipulation[idx,6],
            pre_designed_manipulation[idx,7],pre_designed_manipulation[idx,8],pre_designed_manipulation[idx,9],pre_designed_manipulation[idx,10],pre_designed_manipulation[idx,11],pre_designed_manipulation[idx,12],pre_designed_manipulation[idx,13],
            pre_designed_manipulation[idx,14],pre_designed_manipulation[idx,15],pre_designed_manipulation[idx,16])
    print ("Right hand hand-eye calibrated!")
    print ("========================================================")
    print ("Test proceeding three completes, goes to robotic manipulation...")
    print ("warning: remember to clean garnet explore files...")
###################################################



