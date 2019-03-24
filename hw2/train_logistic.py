import numpy as np
import pandas as pd
import sys

df1 = pd.read_csv(sys.argv[1],encoding = "ISO-8859-1")
df2 = pd.read_csv(str(sys.argv[2]),encoding = "ISO-8859-1")

data = []
ans  = []
for i,row in enumerate(df1.iterrows()):
	index, rowdata = row
	rowdata = rowdata.tolist()
	rowdata.append(1)
	data.append(rowdata)
data = np.array(data,dtype = float)

#normalize
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
std[-1] = 1
mean[-1] = 0
data = (data - mean ) / std

np.save("mean.npy",mean)
np.save("std.npy",std)

# y
for i,row in enumerate(df2.iterrows()):
	index, rowdata = row
	ans.append(rowdata[0])
ans = np.array(ans,dtype = float)


w = [0 for i in range(107)]
w = np.array(w,dtype = float )
lr = 1
time = 10000
gradient_sum = [0 for i in range(107)]
gradient_sum = np.array(gradient_sum,dtype = float )

#adagrad
for i in range(time):
	y = np.dot(data,w)
	y = 1/(1+np.exp(-y)) # sigmoid
	Loss = y - ans
	gradient_w = np.dot(np.transpose(data),Loss) 
	gradient_sum += gradient_w ** 2
	ada = np.sqrt(gradient_sum)
	w -= lr * (gradient_w / ada)
	# if i % 1000 == 999:
	# 	y2 = (y - 1) * -1
	# 	entropy = np.dot(ans , np.log(y+0.000000001)) + np.dot(-1*(ans-1) , np.log(y2+0.000000001))
	# 	print(-entropy)
	# 	print(sum(abs(Loss)))
	# 	print(np.dot(np.round(y,0),ans)/32562)
	# 	print(str(i)+"========================")
np.save("logistic.npy",w)
