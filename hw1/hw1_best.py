import numpy as np
import pandas as pd
import sys

inputfile = sys.argv[1]
outputfile = sys.argv[2]

df = pd.read_csv(inputfile,encoding = "ISO-8859-1",header=None)
w = np.load("para_hw1_best.npy")


data = [[] for i in range(240)]
train_x = []
train_y = []


for i,row in enumerate(df.iterrows()):
	index, rowdata = row
	rowdata = [w.replace('NR', '0') for w in rowdata]
	rowdata = rowdata[2:]
	data[ i//18 ] += rowdata
	if i%18 == 17:
		data[i//18].append(1)



ans = []
data = np.array(data,dtype = float)
for i,each in enumerate(data):
	temp = np.dot(data[i],w)
	if temp < 0:
		temp = 0
	ans.append(["id_"+str(i),temp])
df = pd.DataFrame(ans, columns=["id","value"])
df.to_csv(outputfile,index = False)


