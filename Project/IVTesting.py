import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS
import random
import math
import os
import sys

fsdata = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/fsdlag.csv")
ssdata = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/estimateddifinalpresent.csv")
pricedata = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/cutpdata.csv")

wp = fsdata[["wp","wplag1","wplag2"]]
exogreg = fsdata[["Prod1","Prod2","Prod3","Prod4","Prod5","Prod6","Prod7","Prod8","Prod9","Prod10","Prod11"]]
price = pricedata["price"]

ssdata = ssdata.to_numpy()
sssdata = pd.DataFrame(ssdata[:,1])

#pricedata = pricedata.to_numpy()
#fsdata = fsdata.to_numpy()
model = IV2SLS(dependent = sssdata,exog=exogreg,endog=price,instruments=wp)
results = model.fit(cov_type="robust")
predvalues = results.predict()
original = sssdata
residuals = original.to_numpy() - predvalues.to_numpy()

residuals = pd.DataFrame(residuals)
print(results)
sys.exit()
i = 1

def objfunc(Z,e,etp,ztp,i):
    if i == 0:
        inside = (ztp@Z)
    else:
        inside = (ztp @ e) @ (etp @ Z)
    print("INSIDE")
    print(inside)
    weightingmatrix = 1 / inside
    print(weightingmatrix)
    print(ztp @ e)
    objf = (etp @ Z) @ weightingmatrix @ (ztp @ e)
    print(objf)
    return(objf)

wp = pd.DataFrame(wp)
residtest = residuals[0].values

wptest = wp["wp"].values
wptest = wptest.reshape((10296,1))
residtest = residtest.reshape((10296,1))

rtp = np.transpose(residtest)
wtp = np.transpose(wptest)

objfunc(wptest,residtest,rtp,wtp,1)