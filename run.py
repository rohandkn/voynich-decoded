from voynich import VoynichManuscript
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.legacy.data import Field, TabularDataset, BucketIterator	
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report


class VoynichDataset(Dataset):
	"""Voynich dataset."""

	def tokenizeLine(self, lineText):
		return lineText.split('.')

	def __init__(self):
		self.vm = VoynichManuscript("voynich-text.txt", inline_comments=False)
		self.dataset = []
		self.vocab = set()
		labelList = {}
		self.labelSet = [""] * 6
		# add all page to dataset
		for page in self.vm.pages:
			concatLines = []
			if self.vm.pages[page].section not in labelList:
				self.labelSet[len(labelList)] = self.vm.pages[page].section
				labelList[self.vm.pages[page].section] = len(labelList)
			for line in self.vm.pages[page]:
				self.dataset.append(((line.text.split('.')), labelList[self.vm.pages[page].section], len(line.text.split('.'))))
				self.vocab = self.vocab.union(set(line.text.split('.')))

		# create vocabulary for tokenizing string
		vocab2index = {}
		i = 0
		for elem in self.vocab:
			vocab2index[elem] = i
			i+=1

		for i in range(0, len(self.dataset)):
			tmp = []
			for elem in self.dataset[i][0]:
				tmp.append(vocab2index[elem])
			self.dataset[i] = (torch.FloatTensor(tmp), self.dataset[i][1], self.dataset[i][2])


	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		return self.dataset[idx]

# LSTM module
class LSTM(nn.Module):
	def __init__(self, vocab_len, embed_len, hidden_len):
		super(LSTM, self).__init__()
		self.embedding = nn.Embedding(vocab_len, embed_len, padding_idx=0)
		self.lstm = nn.LSTM(input_size=embed_len, hidden_size=hidden_len, num_layers=1, batch_first=True, bidirectional=True)
		self.dropout = nn.Dropout(0.3)
		self.linear = nn.Linear(hidden_len, 6)
		self.hidden_len = hidden_len
	def forward(self, x, x_len):
		out = self.embedding(x)
		out = self.dropout(out)
		out = pack_padded_sequence(out, x_len, batch_first=True, enforce_sorted=False)
		a, (out, b) = self.lstm(out)
		out = self.linear(out[-1])
		return out

def train_model(model, train_dl, epochs, lr, val_dl, ls):
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	for i in range(epochs):
		model.train()
		sum_loss = 0.0
		total = 0
		print("STARTING EPOCH "+str(i))

		# loop through train dataloader
		for x, y, l in train_dl:
			x = x.long()
			y = y.long()
			y_pred = model(x, l)
			optimizer.zero_grad()
			loss = nn.functional.cross_entropy(y_pred, y)
			loss.backward()
			optimizer.step()
			sum_loss += loss.item()*y.shape[0]
			total += y.shape[0]
		val_loss, val_acc = validation_metrics(model, val_dl, ls)
		if i % 1 == 0:
			print("train loss %.3f, val loss %.3f, val accuracy %.3f" % (sum_loss/total, val_loss, val_acc))


def validation_metrics(model, valid_dl,lS):
	model.eval()
	correct = 0
	total = 0
	sum_loss = 0.0
	sum_rmse = 0.0
	predList = []
	yList = []
	for x, y, l in valid_dl:
		x = x.long()
		y = y.long()
		y_hat = model(x, l)
		loss = nn.functional.cross_entropy(y_hat, y)
		pred = torch.max(y_hat, 1)[1]
		correct += (pred == y).float().sum()
		predList.append(pred)
		yList.append(y)
		total += y.shape[0]
		sum_loss += loss.item()*y.shape[0]
	print(classification_report(yList, predList, target_names=lS))
	return sum_loss/total, correct/total

vm = VoynichManuscript("voynich-text.txt", inline_comments=False)
train = VoynichDataset()
print(len(train))
# create random train and val split
trainD, valD = torch.utils.data.random_split(train, [4589, 800])
dataloader = DataLoader(trainD, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
val_dl1 = DataLoader(valD, batch_size=1, shuffle=True, num_workers=1, drop_last=False)

l = LSTM(len(train.vocab), 300, 20)
print(validation_metrics(l, val_dl1, train.labelSet))
train_model(l, dataloader, 10, .01, val_dl1, train.labelSet)

