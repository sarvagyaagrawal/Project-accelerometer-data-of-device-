import math
from math import sqrt
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

train_data=pd.read_csv("train_time_series.csv")
train_label=pd.read_csv("train_labels.csv")
test_data=pd.read_csv("test_time_series.csv")
test_label=pd.read_csv("test_labels.csv")

def calc_distance(p1,p2):
    distance=np.sqrt(np.sum(np.power(p1-p2,2)))
    #val= distance.item()
    #return val
    return distance

def find_index_train_label(time):
    for i in range(len(train_data)):
        if time==train_data["timestamp"][i]:
            return i
        
def find_index_test_label(time):
    for i in range(len(test_data)):
        if time==test_data["timestamp"][i]:
            return i

def maximum_count(distances):
    #this function accepts a dictionary
    #from the get_neighbours and count maximum classificaton of
    #first k neighbours
##    count=0
##    d_temp={}
##    for i in distances.values():
##        if count<5:
##            if i not in d_temp.keys():
##                d_temp[i]=1
##            else:
##                d_temp[i]+=1
##            count+=1
##    maxi=max(d_temp.values())
##    for j in d_temp.keys():
##        if d_temp[i]==maxi:
##            return i
    #updated function: by using numpy arrays
    count={}
    for i in range(15):
        if distances[i][1] not in count.keys():
            count[distances[i][1]]=1
        else:
            count[distances[i][1]]+=1
    maximum=max(count.values())
    for j in count.keys():
        if count[j]==maximum:
            return j
        
            
#get_neighbours takes test_row from test label
def get_neighbours(test_row):
    #distances={}
    distances=np.empty((len(train_label),2))
    for i in range(len(train_label)):
        pos_train=find_index_train_label(train_label["timestamp"][i])
        pos_test=find_index_test_label(test_row)
        p_test=np.array([test_data["x"][pos_test],test_data["y"][pos_test],test_data["z"][pos_test]])
        p_train=np.array([train_data["x"][pos_train],train_data["y"][pos_train],train_data["z"][pos_train]])
        dist=calc_distance(p_test,p_train)
        #distances[train_label["label"][pos_train]]=dist
        #distances[dist]=train_label["label"][pos_train]
        distances[i][0]=dist
        distances[i][1]=train_label.iloc[i,3]

    #distances=dict(sorted(distances.items(), key=lambda item: item[0]))
    distances=distances[distances[:,0].argsort()]
    #we will classify for first 5 neighbours
    classify=maximum_count(distances)
    return classify

def kNN_classification():
    for test_row in test_label["timestamp"]:
        group=get_neighbours(test_row)
        test_label["label"]=group

kNN_classification()




