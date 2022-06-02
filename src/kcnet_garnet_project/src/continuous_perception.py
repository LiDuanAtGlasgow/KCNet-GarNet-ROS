#type:ignore
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.stats
from shapely import geometry
import time
import pandas as pd
import os

shape_names=['pant','shirt','sweater','towel','tshirt']

def early_stop(embedding,labels,video_labels,confid_circ=None,category_idx=None,video_idx=None):
    n=5
    bandwidth_value=6
    len_video=60
    contours=[]
    color=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd']
    levelled_color=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd']
    standard_points=np.zeros((n,2))
    plt.figure(figsize=(10,10))
    for i in range (n):
        inds=np.where(labels==i+1)[0]
        data=embedding[inds]*10
        x=data[:,0].mean()
        y=data[:,1].mean()
        pdf=scipy.stats.kde.gaussian_kde(data.T)
        q,w=np.meshgrid(range(confid_circ[0],confid_circ[1],1), range(confid_circ[2],confid_circ[3],1))
        r=pdf([q.flatten(),w.flatten()])
        s=scipy.stats.scoreatpercentile(pdf(pdf.resample(1000)), bandwidth_value)
        r.shape=(confid_circ[4],confid_circ[5])
        cont=plt.contour(range(confid_circ[0],confid_circ[1],1), range(confid_circ[2],confid_circ[3],1), r, [s],colors=color[i])
        cont_location=[]
        for line in cont.collections[0].get_paths():
            cont_location.append(line.vertices)
        cont_location=np.array(cont_location)[0]
        contours.append(cont_location)
        plt.plot(x,y,'o',color=color[i],label=shape_names[i])
        standard_points[i,:]=(x,y)

    i=0
    total=0
    inds=np.where(video_labels==video_idx)[0]
    video_data=embedding[inds]*10
    point_count=np.zeros(n)
    pred=100

    for idx in range (len_video):
        i+=1
        x=video_data[:idx+1,0].mean()
        y=video_data[:idx+1,1].mean()
        mean_value_point=np.array([x,y])
        contains_acc=[]
        for m in range (n):
            line = geometry.LineString(contours[m])
            point = geometry.Point(x,y)
            polygon = geometry.Polygon(line)
            if polygon.contains(point):
                contains_acc.append(m)
        if len(contains_acc)==1:
            for select_idx in range (n):
                if contains_acc[0]==select_idx:
                    point_count[select_idx]+=1
        elif len(contains_acc)>1:
            dists=np.zeros((len(contains_acc),2))
            for h in range(len(contains_acc)):
                standard_point=standard_points[contains_acc[h]]
                dis=np.sum(np.power(standard_point-mean_value_point,2))
                dists[h,0]=dis
                dists[h,1]=contains_acc[h]
            min_val=np.argmin(dists[:,0])
            for select_idx in range (n):
                if dists[min_val,1]==select_idx:
                    point_count[select_idx]+=1
        total+=1
        frame_counts=np.zeros(len(point_count))
        for select_idx in range (len(point_count)):
            count_percentage=point_count[select_idx]/total
            frame_counts[select_idx]=count_percentage*100
            if i>=20:
                if count_percentage>=0.8:
                    pred=select_idx
        if i<=12:
            plt.plot(video_data[idx,0],video_data[idx,1],'o',color=levelled_color[0])
        elif 12<i<=24:
            plt.plot(video_data[idx,0],video_data[idx,1],'o',color=levelled_color[1])
        elif 24<i<=36:
            plt.plot(video_data[idx,0],video_data[idx,1],'o',color=levelled_color[2])
        elif 36<i<=48:
            plt.plot(video_data[idx,0],video_data[idx,1],'o',color=levelled_color[3])
        elif 48<i<=60:
            plt.plot(video_data[idx,0],video_data[idx,1],'o',color=levelled_color[4])     
    plt.legend(loc='best')
    plt.show()                   

    if pred<100:
        print ('[ground-truth]',shape_names[category_idx],'[pred]:',shape_names[pred])
        return shape_names[category_idx],shape_names[pred]
    else:
        print ('[ground-truth]',shape_names[category_idx],'[pred]: Failed!')
        return shape_names[category_idx], 'failed'

def early_stop_for_animation_csv(embedding,labels,video_labels,confid_circ=None,category_idx=None,video_idx=None):
    n=5
    bandwidth_value=6
    len_video=60
    contours=[]
    color=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd']
    standard_points=np.zeros((n,2))
    plt.figure(figsize=(10,10))
    csv_address='./garnet_cp_tracking.csv'
    f=open(csv_address,'w')
    csv_writer=csv.writer(f)
    csv_writer.writerow(('no.','pants','shirts','sweaters','towels','t-shirts','accuracy'))
    for i in range (n):
        inds=np.where(labels==i+1)[0]
        data=embedding[inds]*10
        x=data[:,0].mean()
        y=data[:,1].mean()
        pdf=scipy.stats.kde.gaussian_kde(data.T)
        q,w=np.meshgrid(range(confid_circ[0],confid_circ[1],1), range(confid_circ[2],confid_circ[3],1))
        r=pdf([q.flatten(),w.flatten()])
        s=scipy.stats.scoreatpercentile(pdf(pdf.resample(1000)), bandwidth_value)
        r.shape=(confid_circ[4],confid_circ[5])
        cont=plt.contour(range(confid_circ[0],confid_circ[1],1), range(confid_circ[2],confid_circ[3],1), r, [s],colors=color[i])
        cont_location=[]
        for line in cont.collections[0].get_paths():
            cont_location.append(line.vertices)
        cont_location=np.array(cont_location)[0]
        contours.append(cont_location)
        plt.plot(x,y,'o',color=color[i],label=shape_names[i])
        standard_points[i,:]=(x,y)

    i=0
    total=0
    inds=np.where(video_labels==video_idx)[0]
    video_data=embedding[inds]*10
    point_count=np.zeros(n)
    pred=100

    for idx in range (len_video):
        i+=1
        x=video_data[:idx+1,0].mean()
        y=video_data[:idx+1,1].mean()
        mean_value_point=np.array([x,y])
        contains_acc=[]
        for m in range (n):
            line = geometry.LineString(contours[m])
            point = geometry.Point(x,y)
            polygon = geometry.Polygon(line)
            if polygon.contains(point):
                contains_acc.append(m)
        if len(contains_acc)==1:
            for select_idx in range (n):
                if contains_acc[0]==select_idx:
                    point_count[select_idx]+=1
        elif len(contains_acc)>1:
            dists=np.zeros((len(contains_acc),2))
            for h in range(len(contains_acc)):
                standard_point=standard_points[contains_acc[h]]
                dis=np.sum(np.power(standard_point-mean_value_point,2))
                dists[h,0]=dis
                dists[h,1]=contains_acc[h]
            min_val=np.argmin(dists[:,0])
            for select_idx in range (n):
                if dists[min_val,1]==select_idx:
                    point_count[select_idx]+=1
        total+=1
        frame_counts=np.zeros(len(point_count))
        for select_idx in range (len(point_count)):
            count_percentage=point_count[select_idx]/total
            frame_counts[select_idx]=count_percentage*100
            if i>=20:
                if count_percentage>=0.8:
                    pred=select_idx     
        plt.plot(video_data[idx,0],video_data[idx,1],'o',color=color[category_idx])
        if pred<100:
            csv_writer.writerow((i,0,0,0,0,0,pred))
        else:
            csv_writer.writerow((i,frame_counts[0],frame_counts[1],frame_counts[2],frame_counts[3],frame_counts[4],pred))
    plt.legend(loc='best')
    plt.show()                   
    if pred<100:
        print ('[ground-truth]',shape_names[category_idx],'[pred]:',shape_names[pred])
    else:
        print ('[ground-truth]',shape_names[category_idx],'[pred]: Failed!')
    return csv_address

def early_stop_animation(cp_data,samp_number=60):
    file_path='./garnet_cp_animation/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    df=pd.DataFrame({'x':cp_data[0:samp_number,0],'pant':cp_data[0:samp_number,1],'shirt':cp_data[0:samp_number,2],'sweater':cp_data[0:samp_number,3],'towel':cp_data[0:samp_number,4],'tshirt':cp_data[0:samp_number,5]})
    start_time=time.time()
    for t in range (samp_number):
        x=df['x'].to_numpy()[:t+1]
        pant=df['pant'].to_numpy()[:t+1]
        shirt=df['shirt'].to_numpy()[:t+1]
        sweater=df['sweater'].to_numpy()[:t+1]
        towel=df['towel'].to_numpy()[:t+1]
        tshirt=df['tshirt'].to_numpy()[:t+1]
        plt.figure()
        subplot=plt.subplot()
        subplot.plot(x,pant,color='red',label='pant')
        subplot.plot(x,shirt,color='blue',label='shirt')
        subplot.plot(x,sweater,color='green',label='sweater')
        subplot.plot(x,towel,color='purple',label='towel')
        subplot.plot(x,tshirt,color='gray',label='tshirt')
        subplot.set_xlabel('Input frame')
        subplot.set_ylabel('Percentage (%)')
        plt.legend()
        plt.xlim([0,samp_number])
        plt.ylim([0,100])
        plt.savefig('./garnet_cp_animation/frame_'+str(t+1).zfill(4)+'.png')
        plt.close()
        if t%int(samp_number/10)==0:
            print (f'GarNet cp tracking frame [{t+1}/{samp_number}] has been finished, time={time.time()-start_time}')
            start_time=time.time()
