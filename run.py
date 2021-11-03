from voynich import VoynichManuscript
import torch
from torchtext.legacy.data import Field
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

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
	optimizer = torch.optim.Adam(parameters, lr=lr)
	for i in range(epochs):
		model.train()

vm = VoynichManuscript("voynich-text.txt", inline_comments=False)