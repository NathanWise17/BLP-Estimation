import pandas as pd
import numpy as np
import os
#Read in Data
df = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/data.csv")

#Define Key Parameters
num_rows = 9
num_columns = 104
array_size = (11,11)
group_size = 11

#Define Final Arrays
main_array_olsblogit = np.zeros((num_rows,num_columns), dtype= object)
main_array_ivblogit = np.zeros((num_rows,num_columns), dtype= object)
main_array_olsnested = np.zeros((num_rows,num_columns), dtype= object)
main_array_ivnested = np.zeros((num_rows,num_columns), dtype= object)

#Pull in Price and Share Data, Reshape to be defined in markets
bigprices = df.iloc[:,5].to_numpy()
reshaped_prices = bigprices[:num_rows*num_columns*group_size].reshape(num_rows, num_columns, group_size)

bigshares = df.iloc[:,12].to_numpy()
reshaped_shares = bigshares[:num_rows*num_columns*group_size].reshape(num_rows, num_columns, group_size)

bigwithingroupshares = df.iloc[:,32].to_numpy()
reshaped_wgs = bigwithingroupshares[:num_rows*num_columns*group_size].reshape(num_rows, num_columns, group_size)

print(df)
print(bigwithingroupshares)
ownpricebasicols = []
ownpricebasiciv = []
ownpricenestedols = []
ownpricenestediv = []

crossbasicols = [[] for _ in range(11)]
crossbasiciv = [[] for _ in range(11)]
crossnestedols = [[] for _ in range(11)]
crossnestediv = [[] for _ in range(11)]


#Define some temporary arrays
prices = np.zeros((1,11))
shares = np.zeros((1,11))
wgs = np.zeros((1,11))
tempvar = 0

#Define Alpha Values, Estimated in R
abasiclogit = -103.050
aivlogit = -108.623
aolsnested = -43.586
aivnested = -75.313

#Define Sigma Values, Estimated in R
sigmaolsnested = .813188
sigmaivnested = .738393

#Defining Large Arrays
for i in range(num_rows):
    for j in range(num_columns):
        main_array_olsblogit[i,j]=np.zeros(array_size)
        main_array_ivblogit[i,j] = np.zeros(array_size)
        main_array_olsnested[i,j] = np.zeros(array_size)
        main_array_ivnested[i,j] = np.zeros(array_size)

#Basic Logit, OLS Alpha
for i in range(num_rows):
    for j in range(num_columns):
        
        prices = reshaped_prices[i,j]
        shares = reshaped_shares[i,j]
        temparray = np.zeros(array_size)
        

        for k in range(group_size):
            tempavg = []
            avg = 0
            for l in range(group_size):
                if k == l:
                    tempvar = abasiclogit * prices[k] * (1-shares[k])
                    temparray[k,l] = tempvar
                    ownpricebasicols.append(tempvar)
                else: 
                    tempvar = (-1) * abasiclogit * prices[l] * (shares[l])
                    temparray[k,l] = tempvar
                    tempavg.append(tempvar)
            avg = sum(tempavg) / len(tempavg)
            crossbasicols[k].append(avg)
        
        main_array_olsblogit[i,j] = temparray

print("All finished with OLS Logit Elasticities")

#Basic Logit, IV Alpha

for i in range(num_rows):
    for j in range(num_columns):
        
        prices = reshaped_prices[i,j]
        shares = reshaped_shares[i,j]
        temparray = np.zeros(array_size)
        

        for k in range(group_size):
            tempavg = []
            avg = 0
            for l in range(group_size):
                if k == l:
                    tempvar = aivlogit * prices[k] * (1-shares[k])
                    temparray[k,l] = tempvar
                    ownpricebasiciv.append(tempvar)
                else: 
                    tempvar = (-1) * aivlogit * prices[l] * (shares[l])
                    temparray[k,l] = tempvar
                    tempavg.append(tempvar)
            avg = sum(tempavg) / len(tempavg)
            crossbasiciv[k].append(avg)
        
        main_array_olsblogit[i,j] = temparray

print("All finished with IV Basic Logit Elasticities")


print("- - - - - Moving on to Nested Logit - - - - -")

#Define some commonly used terms
firstsigmaols = (1/(1-sigmaolsnested))
secondsigmaols = (sigmaolsnested/(1-sigmaolsnested))

firstsigmaiv = (1/(1-sigmaivnested))
secondsigmaiv = (sigmaivnested/(1-sigmaivnested))

#OLS Nested Logit
for i in range(num_rows):
    for j in range(num_columns):
        
        prices = reshaped_prices[i,j]
        shares = reshaped_shares[i,j]
        wgs = reshaped_wgs[i,j]
        temparray = np.zeros(array_size)

        for k in range(group_size):
            tempavg = []
            avg = 0
            for l in range(group_size):
                if k == l: 
                    tempvar = aolsnested * shares[k] * (firstsigmaols - secondsigmaols*wgs[k] - shares[k]) * (prices[k] / shares[k])
                    temparray[k,l] = tempvar
                    ownpricenestedols.append(tempvar)

                else:
                    tempvar = (-1)*aolsnested * shares[l] * (secondsigmaols * wgs[k] + shares[k])* (prices[l] / shares[k])
                    temparray[k,l] = tempvar
                    tempavg.append(tempvar)
            avg = sum(tempavg) / len(tempavg)
            crossnestedols[k].append(avg)

        
        main_array_olsnested[i,j] = temparray

print("Finished with OLS Nested Logit Elasticities")

#IV Nested Logit
for i in range(num_rows):
    for j in range(num_columns):
        
        prices = reshaped_prices[i,j]
        shares = reshaped_shares[i,j]
        wgs = reshaped_wgs[i,j]
        temparray = np.zeros(array_size)

        for k in range(group_size):
            tempavg = []
            avg = 0
            for l in range(group_size):
                if k == l: 
                    tempvar = aivnested * shares[k] * (firstsigmaiv - secondsigmaiv*wgs[k] - shares[k]) * (prices[k] / shares[k])
                    temparray[k,l] = tempvar
                    ownpricenestediv.append(tempvar)
                else:
                    tempvar = (-1)*aivnested * shares[l] * (secondsigmaiv * wgs[k] + shares[k])* (prices[l] / shares[k])
                    temparray[k,l] = tempvar
                    tempavg.append(tempvar)
            avg = sum(tempavg) / len(tempavg)
            crossnestediv[k].append(avg)
                    
        
        main_array_olsnested[i,j] = temparray

print("Finished with IV Nested Logit Elasticities")
print("- - - - Exporting Elasticities - - - -")
os.chdir("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/Elasticities")
#Own Price
tempdf = pd.DataFrame(ownpricebasicols)
tempdf.to_csv("ownbasicols.csv")

tempdf = pd.DataFrame(ownpricebasiciv)
tempdf.to_csv("ownbasiciv.csv")

tempdf = pd.DataFrame(ownpricenestedols)
tempdf.to_csv("ownnestedols.csv")

tempdf = pd.DataFrame(ownpricenestediv)
tempdf.to_csv("ownnestediv.csv")

#Cross Price
crossbasicols = np.array(crossbasicols).reshape((10296,1))
tempdf = pd.DataFrame(crossbasicols)
tempdf.to_csv("crossbasicols.csv")

crossbasiciv = np.array(crossbasiciv).reshape((10296,1))
tempdf = pd.DataFrame(crossbasiciv)
tempdf.to_csv("crossbasiciv.csv")

crossnestedols = np.array(crossnestedols).reshape((10296,1))
tempdf = pd.DataFrame(crossnestedols)
tempdf.to_csv("crossnestedols.csv")

crossnestediv = np.array(crossnestediv).reshape((10296,1))
tempdf = pd.DataFrame(crossnestediv)
tempdf.to_csv("crossnestediv.csv")


print("All finished with elasticities, have a nice day!")