import glob
import os
import pandas as pd


######################################Reading the Anamoly Values and writing the rows that have steady state values as 1 into a csv file
os.chdir('E:\\Datasets\\Experiments_with_Anomalies')
anamoly_list = []
for file in glob.glob('*.csv'):
    print(file)
    anamoly = pd.read_csv(file, index_col=None, header=0)
    anamoly_list.append(anamoly)
    
all_anamoly = pd.concat(anamoly_list, axis=0, ignore_index=True)
all_anamoly = all_anamoly[all_anamoly.Sds_Armed != 0]
os.chdir('E:\\Datasets')
anamoly_csv = all_anamoly.to_csv (r'merged_exp_contains_anomalies.csv', index = None, header=True)

######################################Reading the Normal Values and writing the rows that have steady state values as 1 into a csv file

os.chdir('E:\\Datasets\\Normal_Experiments')
normal_list = []
for file in glob.glob('*.csv'):
    print(file)
    normal = pd.read_csv(file, index_col=None, header=0)
    normal_list.append(normal)
all_normal = pd.concat(normal_list, axis=0, ignore_index=True)
all_normal = all_normal[all_normal.Sds_Armed != 0]
os.chdir('E:\\Datasets')
normal_csv = all_normal.to_csv (r'merged_exp_normal.csv', index = None, header=True)
