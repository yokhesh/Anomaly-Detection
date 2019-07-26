import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
start_time = time.time()
import os,psutil
#from scipy.spatial import distance

anamoly = pd.read_csv('merged_exp_contains_anomalies.csv')
normal = pd.read_csv('merged_exp_normal.csv')
anamoly = anamoly.drop(['Sds_Armed'], axis=1)
normal = normal.drop(['Sds_Armed'], axis=1)
correct_val = anamoly[anamoly.Anomaly_Tag != 0]
anamoly_val = anamoly[anamoly.Anomaly_Tag != 1]
anamoly_label = pd.DataFrame()
anamoly_label['Anomaly_Tag'] = anamoly['Anomaly_Tag']
anamoly = anamoly.drop(['Anomaly_Tag'], axis=1)
normal = normal.drop(['Anomaly_Tag'], axis=1)
###Feature normalisation

q75_an, q25_an = np.percentile(anamoly, [75 ,25],axis = 0)
iqr_an = q75_an-q25_an
med_an = np.median(anamoly,axis = 0)
anamoly = (anamoly-med_an)/iqr_an

####Deciding the mean for cluster 1
anamoly_ar = np.array(anamoly)
anamoly_ar = np.delete(anamoly_ar,(1,2,3,5,6,7),1)
mu1 = np.mean(anamoly_ar,axis = 0)
d1  = np.linalg.norm((anamoly_ar-mu1),axis = 1)
max_dis = np.argmax(d1)
mu_cl1 = anamoly_ar[max_dis,:]

####Deciding the mean for cluster 2

d2  = np.linalg.norm((anamoly_ar-mu_cl1),axis = 1)
max_dis2 = np.argmax(d2)
mu_cl2 = anamoly_ar[max_dis2,:]

cond = 0
sum = 0

n = np.mean(anamoly_ar,axis = 0)
mu_cl1 = np.max(anamoly_ar,axis = 0)
mu_cl2 = np.min(anamoly_ar,axis = 0)
while(cond==0):
    sum = sum+1
    print(sum)
    ########Calculating distance of each point with mean
    clus1_dis = np.linalg.norm((anamoly_ar-mu_cl1),axis = 1)
    clus2_dis = np.linalg.norm((anamoly_ar-mu_cl2),axis = 1)
    ########Expanding the dimensions
    clus1_dis = np.expand_dims(clus1_dis, axis=1)
    clus2_dis = np.expand_dims(clus2_dis, axis=1)
    ########concatenating the distance of the clusters for comparison
    cluster_dis = np.concatenate((clus1_dis, clus2_dis), axis=1)
    ########Getting the indices of the maximum value 
    max_ind = np.argmax(cluster_dis,axis = 1)
    ########Getting the indices with value 0 and value 1
    ind_c1 = np.where(max_ind == 0)[0]
    ind_c2 = np.where(max_ind == 1)[0]
    ######Updating the cluster
    cluster1= anamoly_ar
    cluster2= anamoly_ar
    cluster1 = np.delete(cluster1,ind_c2,0)
    cluster2 = np.delete(cluster2,ind_c1,0)
    new_mean1 = np.mean(cluster1,axis = 0)
    new_mean2 = np.mean(cluster2,axis = 0)
    ####Equating the mean condition
    me1_eq_bo = np.equal(mu_cl1,new_mean1)
    me1_eq_bi = me1_eq_bo.astype(int)
    me2_eq_bo = np.equal(mu_cl2,new_mean2)
    me2_eq_bi = me2_eq_bo.astype(int)
    me1_dif = np.sum(me1_eq_bi)
    me2_dif = np.sum(me2_eq_bi)
    diff1 = np.sum(np.abs(me1_dif-mu_cl1))
    diff2 = np.sum(np.abs(me2_dif-mu_cl2))
    if ((me1_dif == 2) and (me2_dif == 2)):
        cond = 1
    if sum ==1000:
        cond = 1
    if ((me1_dif != 2) and (me2_dif != 2)):
        res1 = mu_cl1
        res2 = mu_cl2
        mu_cl1 = new_mean1
        mu_cl2 = new_mean2

test_ind = np.where(anamoly_label == 1)[0]
anamolyc1 = np.isin(test_ind, ind_c1).astype(int)
anamolyc1 = np.sum(anamolyc1)
anamolyc2 = np.isin(test_ind, ind_c2).astype(int)
anamolyc2 = np.sum(anamolyc2)


pd.set_option('display.width', 320)
pd.set_option('display.max_columns',10)
metrics = pd.DataFrame({'Heading':['Cluster 1','Cluster 2'],'Number of anomaly points':[anamolyc1,anamolyc2],'Total number of points':[cluster1.shape[0],cluster2.shape[0]]})
print(metrics)
###########################Time Taken Calculation########################
end_time = time.time()
time_taken = end_time - start_time

###########################Memory usage########################

get_pi = os.getpid()
proc = psutil.Process(get_pi)
memoryusage = proc.memory_info()
