import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
df1 = pd.read_csv('./X_train.csv',encoding = "ISO-8859-1")
df2 = pd.read_csv('./Y_train.csv',encoding = "ISO-8859-1")
data0 = []
data1 = []
ans  = []


for i,row in enumerate(df2.iterrows()):
	index, rowdata = row
	ans.append(rowdata[0])
print(sum(ans))

for i,row in enumerate(df1.iterrows()):
	index, rowdata = row
	rowdata = rowdata.tolist()
	if ans[i] == 0:
		data0.append(rowdata)
	else:
		data1.append(rowdata)
data0 = np.array(data0,dtype = float)
data1 = np.array(data1,dtype = float)
print(np.shape(data0))
print(np.shape(data1))

std1 = np.std(data1,axis=0)
mean0 = np.mean(data0, axis=0)
mean1 = np.mean(data1, axis=0)

len0 = np.shape(data0)[0]
len1 = np.shape(data1)[0]

cov_0 = np.cov(np.transpose(data0))
cov_1 = np.cov(np.transpose(data1))
cov =  (cov_0 * len0 + cov_1 * len1)/(len0 + len1)
# print(cov)
cov_inverse = np.linalg.pinv(cov)
w = np.dot((mean0 - mean1),cov_inverse)
print(w) 
b = (-0.5) * np.dot(np.dot(mean0 , cov_inverse),np.transpose(mean0))
b += 0.5 * np.dot(np.dot(mean1 , cov_inverse),np.transpose(mean1))
b += np.log(len0/len1)
print(b)
#cov0 = np.dot(np.transpose(data0 - mean0),data0 - mean0)/np.shape(data0)[0]
#cov1 = np.dot(np.transpose(data1 - mean1),data1 - mean1)/np.shape(data1)[0]
# print(np.shape(cov1))
# print(cov1)
# print(std1**2)
# print(np.shape(cov0))
# print(np.shape(data0)[0])
# cov = (cov0 * np.shape(data0)[0] + cov1 * np.shape(data1)[0])/(np.shape(data0)[0]+np.shape(data1)[0])
# print(cov)
# print(np.shape(mean))



df3 = pd.read_csv('./X_test.csv',encoding = "ISO-8859-1")
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
df.to_csv("6.csv",index = False)

	# test.append(rowdata)
# test = np.array(test,dtype = float)

# ans0 = multivariate_normal.pdf(test, mean=mean0, cov=cov,allow_singular=True)
# ans1 = multivariate_normal.pdf(test, mean=mean1, cov=cov,allow_singular=True)

# print(ans0)
# print(ans1)

# print(np.shape(test))
# # print(data)
# output = []
# for i in range(len(ans0)):
# 	if ans0[i] * len0 < ans1[i] * len1:
# 		temp = 1
# 	else:
# 		temp = 0
# 	output.append([str(i+1),temp])
# df = pd.DataFrame(output, columns=["id","label"])
# df.to_csv("6.csv",index = False)
