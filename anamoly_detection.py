import time
start_time = time.time()
import os,psutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from numpy import linalg as LA
from astropy.stats import median_absolute_deviation
from mpl_toolkits import mplot3d
from sklearn.metrics import precision_score,recall_score, confusion_matrix, classification_report,accuracy_score

pd.set_option('display.width', 320)
pd.set_option('display.max_columns',10)

#############################################Data Loading#############################################
#test = pd.read_csv('test.csv')
#test = test[test.Sds_Armed != 0]
##test  = test.drop(['Sds_Armed'], axis=1)
#test = test.drop(['Anomaly_Tag'], axis=1)


anomaly = pd.read_csv('merged_exp_contains_anomalies.csv')
normal = pd.read_csv('merged_exp_normal.csv')
anomaly = anomaly.drop(['Sds_Armed'], axis=1)
normal = normal.drop(['Sds_Armed'], axis=1)

#########################################Question 3 start############################################
def question3(anomaly):
    corelation = anomaly.corr()
    corelation = corelation.drop(['Anomaly_Tag'],axis = 0)
    corelation_features_to_anomaly = corelation['Anomaly_Tag']
    corelation_features_to_anomaly1 = pd.DataFrame()
    corelation_features_to_anomaly1['Correlation_table'] = corelation_features_to_anomaly
    return corelation_features_to_anomaly1
corelation_features_to_anomaly = question3(anomaly)
#########################################Question 3 end ############################################





##################################################Feature normalisation start########################################
##Considering all features, so removing anomaly tag
anomaly_label = pd.DataFrame()
anomaly_label['Anomaly_Tag'] = anomaly['Anomaly_Tag']
anomaly = anomaly.drop(['Anomaly_Tag'], axis=1)
normal = normal.drop(['Anomaly_Tag'], axis=1)

#anomaly = np.log(anomaly)
#normal = np.log(normal)

###median-mad
mad_an = median_absolute_deviation(anomaly,axis = 0)
mad_nor = median_absolute_deviation(normal,axis = 0)

q75_an, q25_an = np.percentile(anomaly, [75 ,25],axis = 0)
iqr_an = q75_an-q25_an
q75_nor, q25_nor = np.percentile(normal, [75 ,25],axis = 0)
iqr_nor = q75_nor-q25_nor
med_an = np.median(anomaly,axis = 0)
med_nor = np.median(normal,axis = 0)

anomaly = (anomaly-med_an)/iqr_an
normal = (normal-med_nor)/iqr_nor


##################################################Feature normalisation end########################################

#########################################Question 4 start############################################
def question4(normal):
####features are independent. first case - consider all features
    parameter_fit_case1 = pd.DataFrame()
    parameter_fit_case1['value'] = ['Mean', 'Variance'] 
    for i in normal.columns:
        mu_fid1 = (normal[i].sum())/normal.shape[0]
        s2_fid1 = (((normal[i] - mu_fid1)**2).sum())/normal.shape[0]
        parameter_fit_case1[i] = [mu_fid1,s2_fid1]
    
###features are independent. Second case - Mark the features that are important
###We can see from correlation value of each features that X1 and X5 are the most important features
  # So, Marking the features X1,X5
    parameter_fit_case2 = pd.DataFrame()
    parameter_fit_case2['value'] = ['Mean', 'Variance']
    parameter_fit_case2['X1'] = parameter_fit_case1['X1']
    parameter_fit_case2['X5'] = parameter_fit_case1['X5']

###features are independent. Third case - Doing PCA
### Projecting all the features in terms of X1, X5
    mu_fd1 = (normal.sum())/normal.shape[0]
###Calculating covariance
    labels1 = mu_fd1.keys()
    normal_ind = pd.DataFrame()
    for i in range(8):
        normal_ind[labels1[i]] = (normal[labels1[i]] - mu_fd1[labels1[i]])
    normal_indt = normal_ind.T
    cov = (normal_indt.dot(normal_ind))/normal.shape[0]
    
    eig_val, eig_vec = LA.eig(cov)
    imp_f = eig_vec[:,0:2]
    normal_ar = np.array(normal)
    fea_proj_nor = np.dot(normal_ar,imp_f)
    fea_proj_nor = pd.DataFrame(fea_proj_nor)
    parameter_fit_case3 = pd.DataFrame()
    parameter_fit_case3['value'] = ['Mean', 'Variance']
    for i in fea_proj_nor.columns:
            mu_fid3 = (fea_proj_nor[i].sum())/fea_proj_nor.shape[0]
            s2_fid3 = (((fea_proj_nor[i] - mu_fid3)**2).sum())/fea_proj_nor.shape[0]
            parameter_fit_case3[i] = [mu_fid3,s2_fid3]
    return parameter_fit_case1,parameter_fit_case2,parameter_fit_case3,imp_f

#########################################Question 4 end############################################


#########################################Question 5 start############################################
def question5(normal):
####features are dependent. first case - consider all features
    mu_fd1 = (normal.sum())/normal.shape[0]
###Calculating covariance
    labels1 = mu_fd1.keys()
    normal_ind = pd.DataFrame()
    for i in range(8):
        #print(labels1[i])
        normal_ind[labels1[i]] = (normal[labels1[i]] - mu_fd1[labels1[i]])
    normal_indt = normal_ind.T
    cov = (normal_indt.dot(normal_ind))/normal.shape[0]

###features are dependent. Second case - Mark the features that are important
# So, Marking the features X1,X5
    normal_mark = pd.DataFrame()
    normal_mark['X1'] = normal['X1']
    normal_mark['X5'] = normal['X5']
    mu_fd2 = (normal_mark.sum())/normal_mark.shape[0]
###Calculating covariance
    labels2 = mu_fd2.keys()
    normal_ind2 = pd.DataFrame()
    for i in range(2):
        #print(labels2[i])
        normal_ind2[labels2[i]] = (normal_mark[labels2[i]] - mu_fd2[labels2[i]])
    normal_ind2t = normal_ind2.T
    cov2 = (normal_ind2t.dot(normal_ind2))/normal_mark.shape[0]
###########features are dependent. Third case PCA 
    eig_val, eig_vec = LA.eig(cov)
    imp_f = eig_vec[:,0:2]
    normal_ar = np.array(normal)
    fea_proj_nor = np.dot(normal,imp_f)
    fea_proj_nor = pd.DataFrame(fea_proj_nor)
    
    fea_proj_nor1 = pd.DataFrame()
    fep = fea_proj_nor.iloc[:,0]
    fea_proj_nor1['X1'] = fep
    fep1 = fea_proj_nor.iloc[:,1]
    fea_proj_nor1['X5'] = fep1
    
    mu_fd3 = (fea_proj_nor1.sum())/fea_proj_nor1.shape[0]
    #mu_fd3 = np.array(mu_fd3)
    #mu_fd3 = pd.Series(mu_fd3,index=['X1','X5'])
###Calculating covariance
    labels3 = mu_fd3.keys()
    normal_ind3 = pd.DataFrame()
    for i in range(2):
        #print(labels1[i])
        normal_ind3[labels3[i]] = (fea_proj_nor1[labels3[i]] - mu_fd3[labels3[i]])
    normal_ind3t = normal_ind3.T
    cov3= (normal_ind3t.dot(normal_ind3))/fea_proj_nor1.shape[0]
    return mu_fd1,cov,mu_fd2,cov2,mu_fd3,cov3
#########################################Question 5 end############################################


######################################Question 6 independent features start#################################################
def question6ind(parameter_fit_case1,anomaly_label,anomaly,epsilon):
    test_anomaly = anomaly.iloc[0:2000]
    test_tot = test_anomaly
    test_labels = anomaly_label.iloc[0:2000]
    test_labels = test_labels.values.tolist()
    final_list = []
    for k in range(test_tot.shape[0]):
        print(k)
        test = test_tot.iloc[[k]]
        mud1 = parameter_fit_case1.iloc[[0]]
        mud1 = mud1.drop('value',axis= 1)
        test.reset_index(drop=True, inplace=True)
        int1 = (test-mud1)**2
        vard1 = parameter_fit_case1.iloc[[1]]
        vard1 = vard1.drop('value',axis= 1)
        vard1_2 = vard1*2
        int1.reset_index(drop=True, inplace=True)
        vard1_2.reset_index(drop=True, inplace=True)
        int2 = np.exp(-1*(int1/vard1_2))
        int3 = np.sqrt(2*3.14*vard1)
        int2.reset_index(drop=True, inplace=True)
        int3.reset_index(drop=True, inplace=True)
        int4 = int2/int3
        initial = 1
        for i in int4.columns:
            final_prob_d1 = initial*int4[i]
            initial = final_prob_d1
        final_list.append(final_prob_d1)
        initial2 = 1
    pred_bo = (final_list<epsilon)
    pred_bi = pred_bo.astype(int)
    pred_bi=pred_bi.tolist()
    f1s = f1_score(test_labels, pred_bi)
    c= confusion_matrix(test_labels, pred_bi)
    return f1s,c,final_list,test_labels,pred_bi

def pred_best_epsilon(final_list,test_labels):
    step = (np.max(final_list) - np.min(final_list))/1000
    possible_epsilon = np.arange(np.min(final_list),np.max(final_list),step)
    f1_list = []
    for j in range(len(possible_epsilon)):
        pred_bo = (final_list<possible_epsilon[j])
        pred_bi = pred_bo.astype(int)
        pred_bi=pred_bi.tolist()
        f1_list.append(f1_score(test_labels, pred_bi))
    f1_max = np.argmax(f1_list)
    best_epsilon = possible_epsilon[f1_max]
    return best_epsilon,f1_max,f1_list

######################################Question 6 independent features end#################################################

######################################Question 6 Dependent features start#################################################
def question6dep(mu_fd2,cov2,anomaly_label,anomaly,epsilon):
    test_anomaly = anomaly.iloc[0:2000]
    test_tot = test_anomaly
    test_labels = anomaly_label.iloc[0:2000]
    test_labels = test_labels.values.tolist()
    cov_ar = np.array(cov2)
    cov_ar = np.linalg.inv(cov_ar)
    cov_dt =  np.linalg.det(cov2)
    final_li_de = []
    for k in range(test_tot.shape[0]):
        print(k)
        test = test_tot.iloc[[k]]
        int1 = (test-mu_fd2)
        int2 = np.dot(int1, cov_ar)
        int3 = np.dot(int2, int1.T)
        sec_term = np.exp(-0.5*int3)
        fir_term = (2*3.14)**(test.shape[0]/2)
        fir_term1 = fir_term*(cov_dt**0.5)
        p = sec_term/fir_term1
        p = np.asscalar(p)
        final_li_de.append(p)
    pred_bo = (final_li_de<epsilon)
    pred_bi = pred_bo.astype(int)
    pred_bi=pred_bi.tolist()
    f1s = f1_score(test_labels, pred_bi)
    c= confusion_matrix(test_labels, pred_bi)
    return f1s,c,final_li_de,test_labels,pred_bi
######################################Question 6 Dependent features end#################################################
######################################3D plot start #######################################
def plot3(pred_bi,test_labels,anomaly):
    test_labels = np.array(test_labels)
    pred_bi = np.array(pred_bi)
    x1 = (np.array(anomaly['X1']))[0:test_labels.shape[0]]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.plot3D(pred_bi, test_labels,x1, 'gray')
    ax.scatter(pred_bi, test_labels,x1, 'gray')
    ax.set_xlabel('$Prediction$')
    ax.set_ylabel('$True labels$')
    ax.set_zlabel('$X1$')
    plt.show()
    
######################################3D plot stop #######################################

#####NO FUNCTIONS BEYOND THIS POINT. THIS PART IS DEDICATED TO ONLY TO CALLING THE FUNCTION###########
#########################################Calling any necessary functions#####################################################
    
#################################Calling the function for question 4 case 1 , case 2 and case 3 ###################
parameter_fit_case1,parameter_fit_case2,parameter_fit_case3,sig_vec = question4(normal)

#################################Calling the function for question 5 case 1 , case 2 and case 3 ###################
mu_fd1,cov,mu_fd2,cov2,mu_fd3,cov3 = question5(normal)

#################################Calling the function for question 6 independent features case 1###################


##########Uncomment the following lines of code to run the question 6 independent features case 1
#epsilon_q4_c1 =np.array([3.702756563055149e-05]) ########after taking log and converting to gaussian
epsilon_q4_c1 =np.array([1.8418206363005446e-09])
f1s,c,final_list,test_labels,pred_bi = question6ind(parameter_fit_case1,anomaly_label,anomaly,epsilon_q4_c1)
#plot3(pred_bi,test_labels,anomaly)
#best_epsilon,f1_max,f1_list = pred_best_epsilon(final_list,test_labels)
#print(f1_list[f1_max])

#################################Calling the function for question 6 independent features case 2###################
normal_ques4_c2 = pd.DataFrame()
normal_ques4_c2['X2'] = normal['X2']
normal_ques4_c2['X3'] = normal['X3']
anomaly_ques4_c2 = pd.DataFrame()
anomaly_ques4_c2['X1'] = anomaly['X1']
anomaly_ques4_c2['X5'] = anomaly['X5']

##########Uncomment the following lines of code to run the question 6 independent features case 2


epsilon_q4_c2 =np.array([0.014262597713286168])####for the case 2 in question 4

#f1s,c,final_list,test_labels,pred_bi = question6ind(parameter_fit_case2,anomaly_label,anomaly_ques4_c2,epsilon_q4_c2)
#best_epsilon,f1_max,f1_list = pred_best_epsilon(final_list,test_labels)
#plot3(pred_bi,test_labels,anomaly)
#################################Calling the function for question 6 independent features case 3###################

####Feature projection for anomaly data
mu_fd1_anomaly,cov_anomaly,mu_fd2_anomaly,cov2_anomaly,mu_fd3_anomaly,cov3_anomaly  = question5(anomaly)
eig_val_anomaly, eig_vec_anomaly = LA.eig(cov_anomaly)
imp_f_anomaly = eig_vec_anomaly[:,0:2]
anomaly_ar = np.array(anomaly)
fea_proj_anomaly = np.dot(anomaly_ar,imp_f_anomaly)
fea_proj_anomaly = pd.DataFrame(fea_proj_anomaly)


##########Uncomment the following lines of code to run the question 6 independent features case 3

epsilon_q4_c3 = np.array([0.0012206975218819966])################for the case 3 in question 6
#f1s,c,final_list,test_labels,pred_bi = question6ind(parameter_fit_case3,anomaly_label,fea_proj_anomaly,epsilon_q4_c3)
#best_epsilon,f1_max,f1_list = pred_best_epsilon(final_list,test_labels)
#print(f1_list[f1_max])
#plot3(pred_bi,test_labels,anomaly)

#################################Calling the function for question 6 dependent features case 1###################

##########Uncomment the following lines of code to run the question 6 dependent features case 1 function
#epsilon_q5_c1 = np.array([0.03763147397326596])####after taking log and converting to gaussian

epsilon_q5_c1 = np.array([0.0002458266040609138])################for the case 1 in question 5
#f1s,c,final_li_de,test_labels,pred_bi = question6dep(mu_fd1,cov,anomaly_label,anomaly,epsilon_q5_c1)
#best_epsilon,f1_max,f1_list = pred_best_epsilon(final_li_de,test_labels)
#print(f1_list[f1_max])
#plot3(pred_bi,test_labels,anomaly)

#################################Calling the function for question 6 dependent features case 2###################

##########Uncomment the following lines of code to run the question 6 dependent features case 2 function
epsilon_q5_c2 = np.array([0.035483341368924004])################for the case 2 in question 5
#f1s,c,final_li_de,test_labels,pred_bi = question6dep(mu_fd2,cov2,anomaly_label,anomaly_ques4_c2,epsilon_q5_c2)
#best_epsilon,f1_max,f1_list = pred_best_epsilon(final_li_de,test_labels)
#print(f1_list[f1_max])
#plot3(pred_bi,test_labels,anomaly)

#################################Calling the function for question 6 dependent features case 3###################
fea_proj_ano = pd.DataFrame()
fep = fea_proj_anomaly.iloc[:,0]
fea_proj_ano['X1'] = fep
fep1 = fea_proj_anomaly.iloc[:,1]
fea_proj_ano['X5'] = fep1


##########Uncomment the following lines of code to run the question 6 dependent features case 3 function

epsilon_q5_c3 = np.array([0.0030590592218447126])################for the case 3 in question 5
#f1s,c,final_li_de,test_labels,pred_bi = question6dep(mu_fd3,cov3,anomaly_label,fea_proj_ano,epsilon_q5_c3)
#best_epsilon,f1_max,f1_list = pred_best_epsilon(final_li_de,test_labels)
#print(f1_list[f1_max])
#plot3(pred_bi,test_labels,anomaly)

###########################Time Taken Calculation########################
end_time = time.time()
time_taken = end_time - start_time

###########################Memory usage########################

get_pi = os.getpid()
proc = psutil.Process(get_pi)
memoryusage = proc.memory_info()


####test


ac = accuracy_score(test_labels, pred_bi)
recall = recall_score(test_labels, pred_bi)
precision = precision_score(test_labels, pred_bi)
cr = classification_report(test_labels, pred_bi)
metrics = pd.DataFrame({'Heading':['accuracy','recall','precision','F1'],'metrics':[ac,recall,precision,f1s]})

print(metrics)
