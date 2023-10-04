import pandas as pd
import numpy as np

#Read in Data
df = pd.read_csv("C:/Users/wisen/OneDrive/Desktop/Work Stuff/PHD/Second Year/IO/Project/data.csv")

#BASIC LOGIT WITH OLS ESTIMATE
arraybasiclogitols = np.zeros((11,11))

num_rows = 9
num_columns = 104
array_size = (11,11)
group_size = 11

main_array_olsblogit = np.zeros((num_rows,num_columns), dtype= object)
bigprices = df.iloc[:,4].to_numpy()
reshaped_prices = bigprices[:num_rows*num_columns*group_size].reshape(num_rows, num_columns, group_size)
print(reshaped_prices)


for i in range(num_rows):
    for j in range(num_columns):
        main_array_olsblogit[i,j] = np.zeros(array_size)

