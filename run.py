from voynich import VoynichManuscript
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.legacy.data import Field, TabularDataset, BucketIterator	

class VoynichDataset(Dataset):
	"""Custom dataset."""

	def tokenizeLine(self, lineText):
		return lineText.split('.')

	def __init__(self):
		"""
		Args:
		csv_file (string): Path to the csv file with annotations.
		root_dir (string): Directory with all the images.
		transform (callable, optional): Optional transform to be applied
		on a sample.
		"""
		self.vm = VoynichManuscript("voynich-text.txt", inline_comments=False)
		self.dataset = []
		self.vocab = set()
		for page in self.vm.pages:
			concatLines = []
			for line in self.vm.pages[page]:
				concatLines += self.tokenizeLine(line.text)
			self.dataset.append((concatLines, self.vm.pages[page].section))
			self.vocab = self.vocab.union(set(concatLines))
		print(len(self.vocab))


	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		return self.dataset[idx]

class LSTM(nn.Module):
	def __init__(self, vocab_len, embed_len, hidden_len):
		super(LSTM, self).__init__()
		self.embedding = nn.Embedding(vocab_len, embed_len, padding_idx=0)
		self.lstm = nn.LSTM(input_size=embed_len, hidden_size=hidden_len, num_layers=1, batch_first=True, bidirectional=True)
		self.dropout = nn.Dropout(0.3)
		self.linear = nn.Linear(hidden_len, 5)
		self.hidden_len = hidden_len
	def forward(self, x, x_len):
		out = self.embedding(x)
		out = self.dropout(out)
		out = pack_padded_sequence(out, x_len, batch_first=True, enforce_sorted=False)
		a, (out, b) = self.lstm(out)
		out = self.linear(out[-1])
		return out

def train_model(model, epochs, lr):
	parameters = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = torch.optim.Adam(parameters, lr=lr)
	for i in range(epochs):
		model.train()
		sum_loss = 0.0
		total = 0
		for x, y, l in train_dl:
			x = x.long()
			y = y.long()
			y_pred = model(x, l)
			optimizer.zero_grad()
			loss = F.cross_entropy(y_pred, y)
			loss.backward()
			optimizer.step()
			sum_loss += loss.item()*y.shape[0]
			total += y.shape[0]
		val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
		if i % 5 == 1:
			print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))

def validation_metrics(model, valid_dl):
	model.eval()
	correct = 0
	total = 0
	sum_loss = 0.0
	sum_rmse = 0.0
	for x, y, l in valid_dl:
		x = x.long()
		y = y.long()
		y_hat = model(x, l)
		loss = F.cross_entropy(y_hat, y)
		pred = torch.max(y_hat, 1)[1]
		correct += (pred == y).float().sum()
		total += y.shape[0]
		sum_loss += loss.item()*y.shape[0]
		sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
	return sum_loss/total, correct/total, sum_rmse/total

vm = VoynichManuscript("voynich-text.txt", inline_comments=False)
train = VoynichDataset()
