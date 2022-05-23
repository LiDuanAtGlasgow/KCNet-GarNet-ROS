#type:ignore
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.stats
from shapely import geometry

shape_names=['pant','shirt','sweater','towel','tshirt']

def early_stop(embedding,labels,video_labels,confid_circ=None,category_idx=None,video_idx=None):
    n=5
    bandwidth_value=6
    len_video=60
    contours=[]
    standard_points=np.zeros((n,2))
    csv_writer=csv.writer(f)
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
        plt.plot(x,y,'o',color=color[i],label=label_plottings[i])
        standard_points[i,:]=(x,y)
    plt.legend(loc='best')
    plt.show()

    i=0
    total=0
    inds=np.where(video_labels==video_idx)[0]
    video_data=embedding[inds]*10
    point_count=np.zeros(n)
    for idx in range (len_video):
        i+=1
        x=video_data[:idx+1,0].mean()
        y=video_data[:idx+1,1].mean()
        #x=video_data[idx,0]
        #y=video_data[idx,1]
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
        pred=100
        frame_counts=np.zeros(len(point_count))
        for select_idx in range (len(point_count)):
            count_percentage=point_count[select_idx]/total
            frame_counts[select_idx]=count_percentage*100
            if i>=20:
                if count_percentage>=0.8:
                    pred=select_idx
            if pred<100:
                csv_writer.writerow((i,0,0,0,0,0,pred))
            else:
                csv_writer.writerow((i,frame_counts[0],frame_counts[1],frame_counts[2],frame_counts[3],
                frame_counts[4],pred))
            
        i=0
        if pred<100:
            print ('[ground-truth]',shape_names[category_idx],'[pred]:',shape_names[pred])
        else:
            print ('[ground-truth]',shape_names[category_idx],'[pred]: Failed!')
        return shape_names[pred], shape_names[pred]