import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import sys


df1 = pd.read_csv(sys.argv[1],encoding = "ISO-8859-1")
df2 = pd.read_csv(sys.argv[2],encoding = "ISO-8859-1")

data0 = []
data1 = []
ans  = []

#data to two group
for i,row in enumerate(df2.iterrows()):
	index, rowdata = row
	ans.append(rowdata[0])

for i,row in enumerate(df1.iterrows()):
	index, rowdata = row
	rowdata = rowdata.tolist()
	if ans[i] == 0:
		data0.append(rowdata)
	else:
		data1.append(rowdata)


data0 = np.array(data0,dtype = float)
data1 = np.array(data1,dtype = float)
mean0 = np.mean(data0, axis=0)
mean1 = np.mean(data1, axis=0)
len0 = np.shape(data0)[0]
len1 = np.shape(data1)[0]


# calculate covariance
cov_0 = np.cov(np.transpose(data0))
cov_1 = np.cov(np.transpose(data1))
cov =  (cov_0 * len0 + cov_1 * len1)/(len0 + len1)
cov_inverse = np.linalg.pinv(cov)

# calculate w and b
w = np.dot((mean0 - mean1),cov_inverse)
# print(w) 


b = (-0.5) * np.dot(np.dot(mean0 , cov_inverse),np.transpose(mean0))
b += 0.5 * np.dot(np.dot(mean1 , cov_inverse),np.transpose(mean1))
b += np.log(len0/len1)
# print(b)


# predict
df3 = pd.read_csv(sys.argv[3],encoding = "ISO-8859-1")
test=[]
output = []
for i,row in enumerate(df3.iterrows()):
	index, rowdata = row
	rowdata = rowdata.tolist()
	rowdata = np.array(rowdata,dtype = float)
	if np.dot(w,rowdata) + b > 0:
		temp = 0
	else:
		temp = 1
	output.append([str(i+1),temp])
df = pd.DataFrame(output, columns=["id","label"])
df.to_csv(sys.argv[4],index = False)

