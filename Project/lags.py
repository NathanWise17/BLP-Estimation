import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS
import scipy.optimize as opt
import random
import math
import os
import time
import sys

fsdata = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/fsd.csv")
data = np.array(fsdata)
finaldata = pd.DataFrame()
for i in range(9):
    indexstart = i * 1144
    endindex = (i+1) * 1144
    storedata = data[indexstart:endindex,:]
    templag = storedata[:,1]
    lag1 = templag[11:-11]
    lag2 = templag[:-22]
    origlag = storedata[22:,1:]
    lag1 = pd.DataFrame(lag1)
    lag2 = pd.DataFrame(lag2)
    origlag = pd.DataFrame(origlag)
    swldata = pd.concat([origlag,lag1],axis=1)
    swldata = pd.concat([swldata,lag2],axis = 1)
    finaldata = pd.concat([finaldata,swldata],axis=0)
         
finaldata.to_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/fsdlag.csv")


bigdata = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/data.csv")
ndata = np.array(bigdata)
findata = pd.DataFrame()
for i in range(9):
    indexstart = i * 1144
    endindex = (i+1) * 1144
    storedata = ndata[indexstart:endindex,:]
    cutdata = storedata[22:,:]
    print(cutdata.shape)
    cdata = pd.DataFrame(cutdata)
    findata = pd.concat([findata,cdata],axis = 0)
print(findata)
findata.to_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/cutdata.csv")

pricedata = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/ssd.csv")
pdata = np.array(pricedata)
fdata = pd.DataFrame()
for i in range(9):
    indexstart = i * 1144
    endindex = (i+1) * 1144
    storedata = pdata[indexstart:endindex,:]
    cutdata = storedata[22:,:]
    print(cutdata.shape)
    cdata = pd.DataFrame(cutdata)
    fdata = pd.concat([fdata,cdata],axis = 0)

fdata.to_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/cutpdata.csv")