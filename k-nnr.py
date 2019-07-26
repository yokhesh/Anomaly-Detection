import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score, confusion_matrix, classification_report,accuracy_score
import time
start_time = time.time()
import os,psutil

anamoly = pd.read_csv('merged_exp_contains_anomalies.csv')
anamoly = anamoly.drop(['Sds_Armed'], axis=1)

anamoly_label = pd.DataFrame()
anamoly_label['Anomaly_Tag'] = anamoly['Anomaly_Tag']
anamoly = anamoly.drop(['Anomaly_Tag'], axis=1)



anamoly_train = anamoly.iloc[0:30000]
anamoly_test = anamoly.iloc[60000:80000]
anamoly_train_label = anamoly_label.iloc[0:30000]
anamoly_test_label = anamoly_label.iloc[60000:80000]

anamoly_train = np.array(anamoly_train)
anamoly_test = np.array(anamoly_test)
anamoly_train_label = np.array(anamoly_train_label)
anamoly_test_label = np.array(anamoly_test_label)

dist = cdist(anamoly_train, anamoly_test, 'euclidean')
class1 = []
for i in range(dist.shape[1]):
    h = np.argmin(dist[:,i])
    local_c1 = anamoly_train_label[h,0]
    class1.append(local_c1)
c = confusion_matrix(class1, anamoly_test_label)
f1s = f1_score(class1, anamoly_test_label)

ac = accuracy_score(class1, anamoly_test_label)
recall = recall_score(class1, anamoly_test_label)
precision = precision_score(class1, anamoly_test_label)
cr = classification_report(class1, anamoly_test_label)
metrics = pd.DataFrame({'Heading':['accuracy','recall','precision','F1'],'metrics':[ac,recall,precision,f1s]})

print(metrics)
###########################Time Taken Calculation########################
end_time = time.time()
time_taken = end_time - start_time

###########################Memory usage########################

get_pi = os.getpid()
proc = psutil.Process(get_pi)
memoryusage = proc.memory_info()
