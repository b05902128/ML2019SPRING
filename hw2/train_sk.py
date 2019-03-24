import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, linear_model
df1 = pd.read_csv(sys.argv[1],encoding = "ISO-8859-1")
df2 = pd.read_csv(sys.argv[2],encoding = "ISO-8859-1")
df3 = pd.read_csv(sys.argv[3],encoding = "ISO-8859-1")
df4 = pd.read_csv(sys.argv[4],encoding = "ISO-8859-1")
df5 = pd.read_csv(sys.argv[5],encoding = "ISO-8859-1")

#edu_num
edu_num = np.array(df4['education_num'])
edu_num = edu_num.reshape(32561,1)
edu_num_test = np.array(df5['education_num'])
edu_num_test = edu_num_test.reshape(16281,1)


df1 = df1.drop(df1.columns[[i for i in range(-42,-4)]+[i for i in range(15,31)]+[-3,-2,10,45,55]+[-47,-46,-45,-44]], axis=1)
df3 = df3.drop(df3.columns[[i for i in range(-42,-4)]+[i for i in range(15,31)]+[-3,-2,10,45,55]+[-47,-46,-45,-44]], axis=1)

# some feature to x**40
X = np.array(df1,dtype = float)
X = np.concatenate((edu_num,X),axis = 1)
front = X[:,:3]
front = np.concatenate( (front,X[:,4:7]),axis = 1)
for i in range(2,41):
	X = np.concatenate((X,front**i),axis = 1)

#normalize
mean = np.mean(X, axis=0)
std  = np.std(X, axis=0)
X = (X - mean)/std


Y = np.array(df2,dtype = float)
Y = np.squeeze(Y)

# X_train, X_train_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=int(sys.argv[1]))

X_test = np.array(df3,dtype = float)
X_test = np.concatenate((edu_num_test,X_test),axis =1)
front = X_test[:,:3]
front = np.concatenate( (front,X_test[:,4:7]),axis = 1)
for i in range(2,41):
	X_test = np.concatenate((X_test,front**i),axis = 1)
X_test = (X_test - mean)/std



logistic_regr = linear_model.LogisticRegression(C=10, penalty='l2')
logistic_regr.fit(X,Y)
# logistic_regr.fit(X_train,y_train)

ans = logistic_regr.predict(X_test)
ans = np.array(ans,dtype = int)
ans_df = pd.DataFrame({'id':range(1,len(ans)+1),'label':ans})
ans_df.to_csv(sys.argv[6],index = False)
