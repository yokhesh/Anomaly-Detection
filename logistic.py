import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score,recall_score, confusion_matrix, classification_report,accuracy_score


anomaly = pd.read_csv('merged_exp_contains_anomalies.csv')
anomaly = anomaly.drop(['Sds_Armed'], axis=1)

anomaly_label = pd.DataFrame()
anomaly_label['Anomaly_Tag'] = anomaly['Anomaly_Tag']
anomaly_label1 = np.array(anomaly_label)
anomaly = anomaly.drop(['Anomaly_Tag'], axis=1)


###Feature normalisation

q75_an, q25_an = np.percentile(anomaly, [75 ,25],axis = 0)
iqr_an = q75_an-q25_an
med_an = np.median(anomaly,axis = 0)
anomaly = (anomaly-med_an)/iqr_an

###bias
epsilon = 1e-5  
lr = 0.01
anomaly_ar = np.array(anomaly)
anomaly_train = anomaly_ar[1:70001,:]
anomaly_test = anomaly_ar[70001:,:]
anomaly_train_la1 = anomaly_label1[1:70001,:]

anomaly_test_la = anomaly_label1[70001:,:]

ap = np.ones(anomaly_train.shape[0])
ap1 = np.expand_dims(ap, axis=1)
anomaly_train1 = np.append(ap1,anomaly_train,axis=1)

apte = np.ones(anomaly_test.shape[0])
apte1 = np.expand_dims(apte, axis=1)
anomaly_test1 = np.append(apte1,anomaly_test,axis=1)


theta = np.zeros(anomaly_train1.shape[1])
theta = np.expand_dims(theta,axis = 1)


####hypothesis:
def hypothesis(theta,anomaly_train1):
    h = np.dot(anomaly_train1,theta)
    h1 = 1/(1 + np.exp(-1*h))
    return h1

####Cost
def cost1(anomaly_train_la1,h1,anomaly_train1):
    c1 = np.multiply(anomaly_train_la1,np.log(h1+epsilon))
    c2 = np.multiply((1-anomaly_train_la1),np.log(1-h1+epsilon))
    cost = -1*(((np.sum(c1+c2))/anomaly_train1.shape[0]))
    return cost

######Gradient descent
def gradient_descent(anomaly_train_la1,h1,anomaly_train1,lr,theta):
    g1= h1-anomaly_train_la1
    g1 = np.repeat(g1, 9, axis=1)
    g2 = np.multiply(anomaly_train1,g1)
    g3 = np.sum(g2,axis =0)
    g3 = np.expand_dims(g3, axis=1)
    grad = lr*g3
    theta = theta - grad
    return theta


def logistic_reg(theta,anomaly_train1,anomaly_train_la1,lr):
    for i in range(100):
        h1 = hypothesis(theta,anomaly_train1)
        cost = cost1(anomaly_train_la1,h1,anomaly_train1)
        print(cost)
        theta = gradient_descent(anomaly_train_la1,h1,anomaly_train1,lr,theta)
    return theta
theta = logistic_reg(theta,anomaly_train1,anomaly_train_la1,lr)

pred = hypothesis(theta,anomaly_test1)
pred1 = pred

for i in range(pred.shape[0]):
    if pred[i] >= 0.5:
        pred1[i]= 1
    if pred[i] < 0.5:
        pred1[i]= 0
confusionmatrix = confusion_matrix(anomaly_test_la, pred)
f1s = f1_score(anomaly_test_la, pred)

ac = accuracy_score(anomaly_test_la, pred)
recall = recall_score(anomaly_test_la, pred)
precision = precision_score(anomaly_test_la, pred)
cr = classification_report(anomaly_test_la, pred)
metrics = pd.DataFrame({'Heading':['accuracy','recall','precision','F1'],'metrics':[ac,recall,precision,f1s]})

print(metrics)
