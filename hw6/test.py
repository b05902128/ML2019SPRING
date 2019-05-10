import jieba
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
import numpy as np
import sys

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
	inputfile = sys.argv[1]
	dictfile = sys.argv[2]
	outputfile = sys.argv[3]
	df3 = pd.read_csv(inputfile,encoding = "utf-8")
	value = []
	index = []
	word_model = Word2Vec.load("word2vec_100_mincount5.model").wv
	jieba.load_userdict(dictfile)
	device = torch.device('cuda')
	embed_dim = 100
	model = Net(embed_dim)
	model.load_state_dict(torch.load("model.th")['model'])
	model.to(device)
	model.eval()
	for i in range(20000):
		seg_list = list(jieba.cut(df3.iloc[i][1], cut_all=False))
		wordvector = []
		for j in range(60):
			try:
				vector = word_model[ seg_list[j] ] 
			except:
				vector = [0 for k in range(embed_dim)]
			wordvector.append(vector)
		wordvector = np.array(wordvector,dtype = float)
		sentence = wordvector.reshape((60,1,embed_dim))
		sentence = torch.tensor(sentence)
		sentence_cuda = sentence.to(device, dtype=torch.float)
		output = model(sentence_cuda)
		predict = torch.max(output, 1)[1].cpu()
		value.append(predict)
	value = np.array(value,dtype = int)
	ans_df = pd.DataFrame({'id':range(20000),'label':value})
	ans_df.to_csv(outputfile,index = False)