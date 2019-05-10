import jieba
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
import numpy as np
import sys





class MyDataset(Dataset):
	def __init__(self, X,Y,word_model,dim):
		self.word_model = word_model
		self.data_X = X
		self.data_Y = np.array(Y)
		self.dim = dim
	def __len__(self):
		return np.shape(self.data_Y)[0]
	def __getitem__(self, idx):
		label = self.data_Y[idx]
		if label == 1:
			label = np.array([0.0,1.0],dtype = float)
		else:
			label = np.array([1.0,0.0],dtype = float)
		sentence = self.data_X[idx]
		wordvector = []
		for i in range(min(len(sentence),60)):
			try:
				vector = self.word_model[ sentence[i] ] 
			except:
				vector = [np.random.random() for j in range(self.dim)]
			wordvector.append(vector)
		for i in range(len(sentence),60):
			vector = [0 for j in range(self.dim)]
			wordvector.append(vector)
		wordvector = np.array(wordvector,dtype = float)
		return wordvector,label
class Net(nn.Module):
	def __init__(self, embedding_dim, dropout=0.5):
		super(Net, self).__init__()
		self.embedding_dim = embedding_dim
		# nn.Embedding 可以幫我們建立好字典中每個字對應的 vector
		# self.embeddings = nn.Embedding(n_vocab, embedding_dim)
		# LSTM layer，形狀為 (input_size, hidden_size, ...)
		self.lstm = nn.LSTM(embedding_dim, 300,bidirectional = True, dropout=dropout,num_layers=2)
		# Fully-connected layer，把 hidden state 線性轉換成 output
		self.fc = nn.Sequential(
			nn.Linear(1200, 1000),
			nn.LeakyReLU(0.2),
			nn.BatchNorm1d(1000),
			nn.Linear(1000, 100),
			nn.LeakyReLU(0.2),
			nn.BatchNorm1d(100),
			nn.Dropout(0.5),
			nn.Linear(100, 2),
		)
		self.output = nn.Softmax(dim=1)
	def forward(self, seq_in):
		# LSTM 接受的 input 形狀為 (timesteps, batch, features)，
		# 即 (seq_length, batch_size, embedding_dim)
		# 所以先把形狀為 (batch_size, seq_length) 的 input 轉置後，
		# 再把每個 value (char index) 轉成 embedding vector
		# embeddings = self.embeddings(seq_in.t())
		# LSTM 層的 output (lstm_out) 有每個 timestep 出來的結果
		#（也就是每個字進去都會輸出一個 hidden state）
		# 這邊我們取最後一層的結果，即最近一次的結果，來預測下一個字
		lstm_out, _ = self.lstm(seq_in)
		ht1 = lstm_out[-1]
		ht2 = torch.max(lstm_out,0)[0]
		ht3 = torch.mean(lstm_out,0)
		ht = torch.cat((ht2,ht3),1)
		# ht = torch.cat((ht,ht3),1)
		# 線性轉換至 output
		out = self.fc(ht)
		out = self.output(out)
		return out

if __name__ == '__main__':
	inputfile1 = sys.argv[1]
	inputfile2 = sys.argv[2]
	inputfile3 = sys.argv[3]
	dictfile = sys.argv[4]
	df  = pd.read_csv(inputfile1,encoding = "utf-8")
	df2 = pd.read_csv(inputfile2,encoding = "utf-8")
	df3 = pd.read_csv(inputfile3,encoding = "utf-8")



	train_sentence = []
	test_sentence = []
	all_sentence = []
	train_labels = []
	test_labels = []

	best = 100
	jieba.load_userdict(dictfile)
	embed_dim = 100
	lenlist = []
	for i in range(119018):
		seg_list = list(jieba.cut(df.iloc[i][1], cut_all=False))
		length = len(seg_list)
		lenlist.append(length)
		label = df2.iloc[i][1]
		if  i % 10 == 7:
			test_labels.append(label)
			test_sentence.append(seg_list)
		else:
			train_labels.append(label)
			train_sentence.append(seg_list)
		all_sentence.append(seg_list)

	# for i in range(20000):
	# 	seg_list = list(jieba.cut(df3.iloc[i][1], cut_all=False))
	# 	length = len(seg_list)
	# 	lenlist.append(length)
	# 	all_sentence.append(seg_list)
	# lenlist = np.array(lenlist)
	# print(np.mean(lenlist))33
	# print(np.median(lenlist))17
	# print(np.percentile(lenlist,80))44
	# print(np.percentile(lenlist,90))74
	# print(np.percentile(lenlist,95))112
	# word_model = Word2Vec(all_sentence,size = embed_dim, min_count=5)
	# word_model.save("word2vec_300_mincount5.model")
	word_model = Word2Vec.load("word2vec_100_mincount5.model").wv
	print("OKOK")
	device = torch.device('cuda')

	model = Net(embed_dim)
	model.to(device)
	optimizer = Adam(model.parameters(), lr=0.0001)
	
	loss_fn = nn.MSELoss()
	Traindataset = MyDataset(train_sentence,train_labels,word_model,embed_dim)
	Traindataloader = DataLoader(Traindataset, batch_size=300, shuffle=True, num_workers=4)
	Testdataset = MyDataset(test_sentence,test_labels,word_model,embed_dim)
	Testdataloader = DataLoader(Testdataset, batch_size=300, shuffle=True, num_workers=4)
	for epoch in range(25):
			print("epoch"+str(epoch))
			train_loss = []
			train_acc = []
			model.train()
			for _, (sentence, target) in enumerate(Traindataloader):
				sentence = np.transpose(sentence,(1,0,2))
				sentence = torch.tensor(sentence)
				target = torch.tensor(target)
				sentence_cuda = sentence.to(device, dtype=torch.float)
				target_cuda = target.to(device, dtype=torch.float)
				optimizer.zero_grad()
				output = model(sentence_cuda)
				loss = loss_fn(output, target_cuda)
				loss.backward()
				optimizer.step()
				predict = torch.max(output, 1)[1]
				acc = np.mean((torch.max(target_cuda,1)[1] == predict).cpu().numpy())
				train_acc.append(acc)
				train_loss.append(loss.item())
			print("TrainEpoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))


			model.eval()
			valid_loss = []
			valid_acc = []
			for _, (sentence, target) in enumerate(Testdataloader):
				sentence = np.transpose(sentence,(1,0,2))
				sentence = torch.tensor(sentence)
				target = torch.tensor(target)
				sentence_cuda = sentence.to(device, dtype=torch.float)
				target_cuda = target.to(device, dtype=torch.float)
				output = model(sentence_cuda)
				predict = torch.max(output, 1)[1]
				acc = np.mean((torch.max(target_cuda,1)[1] == predict).cpu().numpy())
				loss = loss_fn(output, target_cuda)
				valid_acc.append(acc)
				valid_loss.append(loss.item())
			if best > np.mean(valid_loss):
				best = np.mean(valid_loss)
				torch.save({'model':model.state_dict()},"best_train_model.th")
			print("validEpoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(epoch + 1, np.mean(valid_loss), np.mean(valid_acc)))
	torch.save({'model':model.state_dict()},"train_model.th")