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



np.save("para_hw1.npy",w)





