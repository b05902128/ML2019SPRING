import numpy as np
import pandas as pd
df = pd.read_csv('./test.csv',encoding = "ISO-8859-1",header=None)
f = open("para.txt","r")
w = [float(each) for each in f.readline().split()]
# b = 0.0017079693463325908



data = [[] for i in range(240)]
train_x = []
train_y = []

for i,row in enumerate(df.iterrows()):
	index, rowdata = row
	rowdata = [w.replace('NR', '0') for w in rowdata]
	data[ i//18 ] += rowdata[2:]
	if i%18 == 17:
		data[i//18].append(1)
print(data[0])
print(len(data[0]))

ans = []
data = np.array(data,dtype = float)
for i,each in enumerate(data):
	# print(np.dot(data[i],w)+b)
	ans.append(["id_"+str(i),np.dot(data[i],w)])
df = pd.DataFrame(ans, columns=["id","value"])
df.to_csv('ans.csv',index = False)


