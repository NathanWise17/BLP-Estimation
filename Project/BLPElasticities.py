import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS
import scipy.optimize as opt
import random
import math
import os
import time
import sys

#Define other stuff
numsamples = 20
numproducts = 11
numstores = 9
numweeks = 102
nummarkets = numstores * numweeks

#DEFINE ESTIMATES
Prod1    =   3.219362
Prod2    =   2.124138
Prod3   =   3.534572
Prod4    =   2.864995
Prod5    =   2.472084
Prod6    =   3.911657
Prod8    =   2.837996
Prod9   =    3.213221
Prod11 =     3.267972
price =   -862.867559
pitest = 368.8384341934952
sigmatest = 315.5121186952357  



#READ IN DATA
testdi = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/estimateddifinalpresent.csv")
inc = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/ifinal.csv")
nu = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/nfinal.csv")
shares = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/eshares.csv")
pricedata = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/cutpdata.csv")

price1 = pricedata["price"]
priced = price1.to_numpy()
priced = priced.reshape((11,918))

delta_0 = testdi.iloc[:,1].to_numpy()
delta_0 = delta_0.reshape((11,918))

#Make Alphai
inc = inc.to_numpy()
inc = inc[:,1:]
nu = nu.to_numpy()
nu = nu[:,1:]

alphai = np.zeros((918,20))
for i in range(nummarkets):
    for j in range(numsamples):
        alphai[i,j] = price + pitest*inc[i,j] + sigmatest*nu[i,j]

alphaipd = pd.DataFrame(alphai)
os.chdir("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/Elasticities")
alphaipd.to_csv("alphai.csv")



isharestemp = np.zeros((nummarkets,numsamples,numproducts))
ishares = np.zeros((nummarkets,numsamples,numproducts))
avg_share_array = np.zeros((numproducts,nummarkets))

ownpricercelas = []
crossrcelas = [[] for _ in range(11)]



for i in range(nummarkets):
    for ns in range(numsamples):
        Sumshare = 0
        Denom = 0
        for pr in range(numproducts):
            isharestemp[i,ns,pr] = np.exp(delta_0[pr,i] + priced[pr,i]*(pitest*inc[i,ns]+sigmatest*nu[i,ns]))
            Sumshare = Sumshare + isharestemp[i,ns,pr]
        Denom = 1 + Sumshare
        for pr in range(numproducts):
            itemp = isharestemp[i,ns,pr]
            ishares[i,ns,pr] = itemp / Denom
    shareavg = np.mean(ishares,axis = 1)
    for pr in range(numproducts):
        avg_share_array[pr,i] = shareavg[i,pr]


elas = np.zeros((11,11))
finnum = np.zeros((11,11))
for i in range(nummarkets):
    tempelasm = np.zeros((11,11))
    tempnums = np.zeros((11,11))
    for j in range(numproducts):
        tempavg = []
        avg = 0 
        for k in range(numproducts):
            temp = 0
            tempelas = 0
            sum = 0
            for ns in range(numsamples):
                if j == k:
                    temp = alphai[i,ns]*ishares[i,ns,j]*(1-ishares[i,ns,j])

                    sum = sum + temp

                else:
                    temp = alphai[i,ns]*ishares[i,ns,j]*ishares[i,ns,k]
                    sum = sum + temp
                  
            if j == k: 
                tempelas = (priced[j,i]/avg_share_array[j,i])*(sum/20)
                tempnum = sum/20
                ownpricercelas.append(tempelas)
            else:
                tempelas = -1*(priced[k,i]/avg_share_array[j,i])*(sum/20)
                tempnum = sum/20
                tempavg.append(tempelas)
            tempelasm[j,k] = tempelas
            tempnums[j,k] = tempnum
        avg = np.sum(tempavg) / len(tempavg)
        crossrcelas[j].append(avg)    
            

    elas = np.concatenate((elas,tempelasm))
    finnum = np.concatenate((finnum,tempnums))

finnum = finnum[11:]
elas = elas[11:]

finnumpd = pd.DataFrame(finnum)
elaspd = pd.DataFrame(elas)
os.chdir("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/Elasticities")
elaspd.to_csv("RCelasticities.csv")
finnumpd.to_csv("estderiv.csv")

tempdf = pd.DataFrame(ownpricercelas)
tempdf.to_csv("ownrc.csv")

crossrcelas = np.array(crossrcelas).reshape((10098,1))
tempdf = pd.DataFrame(crossrcelas)
tempdf.to_csv("crossrc.csv")

