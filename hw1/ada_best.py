import numpy as np
import pandas as pd



df = pd.read_csv('./train.csv',encoding = "ISO-8859-1")
data = [[] for i in range(18)]
train_x = []
train_y = []

for i,row in enumerate(df.iterrows()):
	index, rowdata = row
	data[ i%18 ] += rowdata[3:].tolist()

for month in range(12):
	for hr in range(471):
		train_y.append(data[9][480*month+hr+9])
		temp = []
		for dim in range(18):
			temp += data[dim][480*month+hr:480*month+hr+9]
		temp = [w.replace('NR', '0') for w in temp]
		temp.append(1)
		train_x.append(temp)


train_x = np.array(train_x,dtype = float)
train_y = np.array(train_y,dtype = float)

w = [0 for i in range(163)]
lr = 1
time = 100000
for i in range(time):
	y = np.dot(train_x,w)
	Loss = y - train_y
	gradient_w = 2*np.dot(np.transpose(train_x),Loss)
	w -= lr * gradient_w


print(w)
print(Loss[0:30])
avg = sum(abs(Loss))/len(Loss)
print(avg)
avg = sum(np.array(Loss)**2)/len(Loss)
print(avg)
print(np.sqrt(avg))



np.savetxt("para.txt", w, newline=" ")

