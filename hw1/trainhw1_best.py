import numpy as np
import pandas as pd



df = pd.read_csv('./train.csv',encoding = "ISO-8859-1")
data = [[] for i in range(18)]
train_x = []
train_y = []


for i,row in enumerate(df.iterrows()):
	index, rowdata = row
	rowdata = [w.replace('NR', '0') for w in rowdata]
	data[ i%18 ] += rowdata[3:]
x = np.array(data,dtype = float)
max = np.mean(x,1) + 2 * np.std(x,1)
min = np.mean(x,1) - 2 * np.std(x,1)
min[9] = 0
max[9] = 500


for i in range(18):
	for j in range(len(x[i])):
		if x[i][j] > max[i]:
			x[i][j]=max[i]
		elif x[i][j] < min[i]:
			x[i][j]=min[i]

for month in range(12):
	for hr in range(471):
		train_y.append(x[9][480*month+hr+9])
		temp = []
		for dim in range(18):
			temp += (x[dim][480*month+hr:480*month+hr+9].tolist())
		temp.append(1)
		train_x.append(temp)


train_x = np.array(train_x,dtype = float)
train_y = np.array(train_y,dtype = float)
w = [0 for i in range(163)]
f = open("record.txt","w+")

lr = 1
time = 100000
gradient_sum = 0
for i in range(time):
	y = np.dot(train_x,w)
	Loss = y - train_y
	gradient_w = 2*np.dot(np.transpose(train_x),Loss) 
	gradient_sum += gradient_w ** 2
	ada = np.sqrt(gradient_sum)
	w -= lr * (gradient_w / ada)



np.save("para_hw1_best.npy",w)

