import numpy as np
import pandas as pd
import sys

df1 = pd.read_csv(sys.argv[1],encoding = "ISO-8859-1")
data = []

for i,row in enumerate(df1.iterrows()):
	index, rowdata = row
	rowdata = rowdata.tolist()
	rowdata.append(1)
	data.append(rowdata)

#normalize
mean = np.load("mean.npy")
std = np.load("std.npy")
data = np.array(data,dtype = int)
data = (data - mean)/std

#predict
w = np.load("logistic.npy")
ans = np.dot(data,w)
ans = 1/(1+np.exp(-ans)) # sigmoid
output = []
for i,each in enumerate(ans):
	if each < 0.5:
		temp = 0
	else:
		temp = 1
	output.append([str(i+1),temp])
df = pd.DataFrame(output, columns=["id","label"])
df.to_csv(sys.argv[2],index = False)
