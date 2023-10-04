import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS
import scipy.optimize as opt
import random
import math
import os
import time
import sys

starttime = time.time()
seed_value = 42
np.random.seed(seed_value)
rng = np.random.default_rng(42)



df = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/cutdata.csv")
demodf = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/demo_stores.csv",header=None)
fsdata = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/fsdlag.csv")
pricedata = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/cutpdata.csv")


data = df.to_numpy() 
demodata = demodf.to_numpy()
demodata = demodata[:,1:]

numsamples = 20
numproducts = 11
numstores = 9
numweeks = 102
nummarkets = numstores * numweeks

shares = df.iloc[:,12].to_numpy()
shares = shares.reshape((10098,1))
draw_array = np.empty((9,2), dtype= object)


m = demodata[:,0]
v = demodata[:,1]
v2 = np.multiply(v,v)
s1 = np.multiply(m,m)

mu = np.log(s1 / np.sqrt(v2 + s1))
sigma = np.sqrt(np.log((v2/s1)+1))

for i in range(9):
    for j in range(2):
        
        if j == 0:
            tempsamples = rng.lognormal(mean = mu[i], sigma = sigma[i],size = numsamples)
            tempsamples = tempsamples 
            draw_array[i,j] = tempsamples
        else:
            tempsamples = np.random.randn(numsamples)
            draw_array[i,j] = tempsamples

delta_0 = df.iloc[:,29].to_numpy().reshape((11,918))

incomedraw = draw_array[:,0]
incomedraw = np.vstack(incomedraw)
nudraw = draw_array[:,1]
nudraw = np.vstack(nudraw)

repeated_rows = []
repeated_rows2 = []

for row in incomedraw:
    repeated_rows.extend([row] * 102)
incomedrawfinal = np.vstack(repeated_rows)

for row in nudraw:
    repeated_rows2.extend([row] * 102)
nudrawfinal = np.vstack(repeated_rows2)

tempshare = np.zeros(numsamples)
indshare = np.zeros(numsamples)


avgshare = 0

columnstokeep = [5,13]

price = df.iloc[:,5]
price = price.to_numpy().reshape((11,918))

i = 0
#initguess = (0.3789311359446262,-0.9280313432773832)
initguess = (0.001,0.0001)

wp = fsdata[["wp","wplag1","wplag2"]]
#wp = fsdata[["wp"]]
fulliv = fsdata[["wp","wplag1","wplag2","Prod1","Prod2","Prod3","Prod4","Prod5","Prod6","Prod7","Prod8","Prod9","Prod10","Prod11"]]
#fulliv = fsdata[["wp"]]
exogreg = fsdata[["Prod1","Prod2","Prod3","Prod4","Prod5","Prod6","Prod7","Prod8","Prod9","Prod10","Prod11"]]
price1 = pricedata["price"]
bigprices = price1.values
bigprices = bigprices.reshape((10098,1))
bigexogreg = exogreg.values
i = 0
#weightingmatrix = [[ 1.39431746e+18, -1.84423890e+17,  3.85299186e+17],
                   #[ 1.15292150e+18, -1.15292150e+18,  6.06823329e+17],
                   #[ 0.00000000e+00,  2.88230376e+17, -8.30413219e+16]]

#weightingmatrix = np.array(weightingmatrix)
weightingmatrix = np.identity(14)
#weightingmatrix = 5.06100219e+18
def objfunc(guess,delta_0, price, incomedrawfinal, nudrawfinal,numproducts,nummarkets,numsamples,shares,exogreg,price1,wp,bigexogreg,bigprices,fulliv,weightingmatrix):
    pi_0, sigma_0 = guess
    
    print("NEW GUESSES")
    print(pi_0)
    print(sigma_0)
    print("END NEW GUESSES")
    
    delta_1 = delta_0

    di = contractmap(delta_1,pi_0,sigma_0,price,incomedrawfinal,nudrawfinal,numsamples,numproducts,nummarkets,shares)
    

    model = IV2SLS(dependent = di,exog=exogreg,endog=price1,instruments=wp)
    results = model.fit(cov_type="robust")
    #print(results.params)
    resultsarray = np.array(results.params)
    resultsarray = resultsarray.reshape((12,1))

    #print(resultsarray)
    #print(resultsarray.shape)
    print("Price Alpha Estimate: ",resultsarray[11,0])
    #residtest1 = np.zeros((10296,1))
    #for i in range(10296):
        #xvar = resultsarray[0,0]*bigexogreg[i,0] + resultsarray[1,0]*bigexogreg[i,1] + resultsarray[2,0]*bigexogreg[i,2] + resultsarray[3,0]*bigexogreg[i,3] + resultsarray[4,0]*bigexogreg[i,4] + resultsarray[5,0]*bigexogreg[i,5] + resultsarray[6,0]*bigexogreg[i,6] + resultsarray[7,0]*bigexogreg[i,7] + resultsarray[8,0]*bigexogreg[i,8] + resultsarray[9,0]*bigexogreg[i,9] + resultsarray[10,0]*bigexogreg[i,10]
        #residtest1[i,0] = di[i,0] - (bigprices[i,0]*resultsarray[11,0] + xvar)

    
    predvalues = results.predict()
    original = di
    residuals = original - predvalues.to_numpy()

    residuals = pd.DataFrame(residuals)
    
    fulliv = pd.DataFrame(fulliv)
    residtest = residuals[0].values
    wptest = fulliv.values

     
    
    e = residtest.reshape((10098,1))
    Z = wptest.reshape((10098,14))

    etp = np.transpose(residtest)
    ztp = np.transpose(wptest)
    
    
    #firsthalf = ztp@e
    #secondhalf = etp@Z
    #firsthalf = firsthalf.reshape((14,1))
    #secondhalf = secondhalf.reshape((1,14))
    #inside = firsthalf @ secondhalf
    #print("INSIDE")
    #print(inside)
    #inverse = np.linalg.inv(inside)
    #print("INVERSE")
    #print(inverse)
    #sys.exit()

    objf = (etp @ Z) @ weightingmatrix @ (ztp @ e)
    
    print("The End is Never the End is Never the End is Never the End is Never the End is Never...")
    print("OBJECTIVE VALUE IS:")
    print(objf)
    return(objf)


def diter(delta_0,pi_0,sigma_0,price,incomedraw,nudraw,numsamples,numproducts,nummarkets):
    avg_share_array = np.zeros((11,918))
    for i in range(nummarkets):
        FinalShare = np.zeros((numproducts,numsamples))
        shareavg = 0
        for k in range(numsamples):
            Sumshare = 0
            Indshare = np.zeros((11,1))
            Denom = 0
            for j in range(numproducts):
                Indshare[j,0] = (np.exp(delta_0[j,i] + price[j,i]*(pi_0*incomedraw[i,k]+sigma_0*nudraw[i,k])))
                Sumshare = Sumshare + Indshare[j]
            Denom = 1 + Sumshare
            for j in range(numproducts):
                itemp = Indshare[j,0]
                FinalShare[j,k] = itemp / Denom[0]
        shareavg = np.mean(FinalShare,axis = 1)
        for j in range(numproducts):
            avg_share_array[j,i] = shareavg[j]

    return avg_share_array


def contractmap(delta_0,pi_0,sigma_0,price,incomedrawfinal,nudrawfinal,numsamples,numproducts,nummarkets,shares):
    d0 = delta_0.reshape((11,918))
    avg_shape_array = diter(d0,pi_0,sigma_0,price,incomedrawfinal,nudrawfinal,numsamples,numproducts,nummarkets)
    avg_share_arrayre = avg_shape_array.reshape((10098,1))
    d0 = d0.reshape((10098,1))
    di = d0 + np.log(shares) -np.log(avg_share_arrayre)
    dist = np.linalg.norm(di - d0)
    t = 0
    while abs(dist) > 1e-12:
        d0 = di.reshape((11,918))
        avg_shape_array = diter(d0,pi_0,sigma_0,price,incomedrawfinal,nudrawfinal,numsamples,numproducts,nummarkets)
        avg_share_arrayre = avg_shape_array.reshape((10098,1))
        d0 = d0.reshape((10098,1))
        di = d0 + np.log(shares) -np.log(avg_share_arrayre)
        dist = np.linalg.norm(di-d0)
        t= t+1
    print("Contraction Mapping Iteration: ",t)
    di = di.reshape((10098,1))   
    return(di)





#debug = diter(delta_0,pi_0,sigma_0,price,incomedrawfinal,nudrawfinal,numsamples,nummarkets,numproducts)
#start = -1.0
#stop = 1.0
#step = 0.1
#result_grid = np.zeros((10,10))
#for i in range(int(start/step),int(stop/step)):
#print(result_grid)





giveup = opt.minimize(objfunc,initguess,args = (delta_0, price, incomedrawfinal, nudrawfinal,numproducts,nummarkets,numsamples,shares,exogreg,price1,wp,bigexogreg,bigprices,fulliv,weightingmatrix), method = "Nelder-Mead")
r = 1
#while j > .00001:
    #r = r+1
    #print("STARTING NEXT ROUND: ", r)
    #initguess = giveup.x
    #giveup = opt.minimize(objfunc,initguess,args = (delta_0, price, incomedrawfinal, nudrawfinal,numproducts,nummarkets,numsamples,shares,exogreg,price1,wp,bigexogreg,bigprices), method = "BFGS")
    #j = giveup.fun
    #initguess = giveup.x
    
    

print(giveup.message)
print(giveup.fun)
print(giveup.x)
finalguesspi, finalguesssigma = giveup.x
difinal = contractmap(delta_0,finalguesspi,finalguesssigma,price,incomedrawfinal,nudrawfinal,numsamples,numproducts,nummarkets,shares)

finalmodel = IV2SLS(dependent = difinal,exog=exogreg,endog=price1,instruments=wp)
finalresults = finalmodel.fit(cov_type="robust")
print("FINAL MODEL RESULTS:")
print(finalresults.params)





endtime = time.time()
dipd = pd.DataFrame(difinal)
dipd.to_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/estimateddifinal.csv")
print("Total Time is: ", endtime - starttime)
print("- - - - Theoretically Finished Successfully - - - -")
