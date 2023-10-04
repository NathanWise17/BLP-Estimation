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

#Let's define some ownership matrices

monop = np.ones((11,11))

single = np.identity(11)

multi = [[1,1,0,0,0,0,0,0,0,0,0],
         [1,1,0,0,0,0,0,0,0,0,0],
         [0,0,1,1,1,1,1,1,1,0,0],
         [0,0,1,1,1,1,1,1,1,0,0],
         [0,0,1,1,1,1,1,1,1,0,0],
         [0,0,1,1,1,1,1,1,1,0,0],
         [0,0,1,1,1,1,1,1,1,0,0],
         [0,0,1,1,1,1,1,1,1,0,0],
         [0,0,1,1,1,1,1,1,1,0,0],
         [0,0,0,0,0,0,0,0,0,1,1],
         [0,0,0,0,0,0,0,0,0,1,1]]
multi = np.array(multi)

derivs = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/Elasticities/RCelasticities.csv")
df = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/cutdata.csv")



derivs = derivs.iloc[:,1:].to_numpy()

derivs = derivs.reshape((918,11,11))


price = df.iloc[:,5].to_numpy()
price = price.reshape((918,11))
shares = df.iloc[:,12].to_numpy()
shares = shares.reshape((918,11))

fullmumonop = []
fullmumulti = []
fullmusingle = []

fullmcmonop = []
fullmcsingle = []
fullmcmulti = []

fullmumtest = []
fullmustest = []
fullmumutest = []

for i in range(nummarkets):
    print("Current Market: ",i)
    omegamonop = monop * derivs[i]
    omegasingle = single * derivs[i]
    omegamulti = multi * derivs[i]
    
    omegamonop = omegamonop * -1
    omegasingle = omegasingle * -1
    omegamulti = omegamulti * -1
    print(omegamonop)
    omonopinv = np.linalg.inv(omegamonop)
    osingleinv = np.linalg.inv(omegasingle)
    omultiinv = np.linalg.inv(omegamulti)

    mcmonop = price[i] - omonopinv @ shares[i]
    mcsingle = price[i] - osingleinv @ shares[i]
    mcmulti = price[i] - omultiinv @ shares[i]

    pricemonop = mcmulti + omonopinv @ shares[i]
    pricesingle = mcmulti + osingleinv @ shares[i]
    pricemulti = mcmulti + omultiinv @ shares[i]

    mumtest = pricemonop - mcmulti
    mustest = pricesingle - mcmulti
    mumutest = pricemulti - mcmulti

    mumonop = price[i] - mcmonop
    musingle = price[i] - mcsingle
    mumulti = price[i] - mcmulti

    fullmumonop.append(mumonop)
    fullmusingle.append(musingle)
    fullmumulti.append(mumulti)

    fullmcmonop.append(mcmonop)
    fullmcsingle.append(mcsingle)
    fullmcmulti.append(mcmulti)

    fullmumtest.append(mumtest)
    fullmustest.append(mustest)
    fullmumutest.append(mumutest)


fullmumonop = np.array(fullmumonop)
fullmumonop = fullmumonop.reshape((10098,1))

fullmusingle = np.array(fullmusingle)
fullmusingle = fullmusingle.reshape((10098,1))

fullmumulti = np.array(fullmumulti)
fullmumulti = fullmumulti.reshape((10098,1))

fullmcmonop = np.array(fullmcmonop)
fullmcmonop = fullmcmonop.reshape((10098,1))

fullmcsingle = np.array(fullmcsingle)
fullmcsingle = fullmcsingle.reshape((10098,1))

fullmcmulti = np.array(fullmcmulti)
fullmcmulti = fullmcmulti.reshape((10098,1))

fullmumtest = np.array(fullmumtest).reshape((10098,1))
fullmustest = np.array(fullmustest).reshape((10098,1))
fullmumutest = np.array(fullmumutest).reshape((10098,1))

os.chdir("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/Markups")

temppd = pd.DataFrame(fullmumonop)
temppd.to_csv("mumonop.csv")

temppd = pd.DataFrame(fullmusingle)
temppd.to_csv("musingle.csv")

temppd = pd.DataFrame(fullmumulti)
temppd.to_csv("mumulti.csv")

temppd = pd.DataFrame(fullmcmonop)
temppd.to_csv("mcmonop.csv")

temppd = pd.DataFrame(fullmcsingle)
temppd.to_csv("mcsingle.csv")

temppd = pd.DataFrame(fullmcmulti)
temppd.to_csv("mcmulti.csv")

temppd = pd.DataFrame(fullmumtest)
temppd.to_csv("mumtest.csv")

temppd = pd.DataFrame(fullmustest)
temppd.to_csv("mustest.csv")

temppd = pd.DataFrame(fullmumutest)
temppd.to_csv("mumutest.csv")



print("Task Finished Succesfully")