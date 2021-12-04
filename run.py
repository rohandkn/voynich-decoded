from voynich import VoynichManuscript
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.legacy.data import Field, TabularDataset, BucketIterator	
from sklearn.metrics import classification_report
import numpy as np

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
		self.labelSet = [""]*6
		labelList = {}
		for page in self.vm.pages:
			concatLines = []
			for line in self.vm.pages[page]:
				concatLines += self.tokenizeLine(line.text)
			if self.vm.pages[page].section not in labelList:
				self.labelSet[len(labelList)] = self.vm.pages[page].section
				labelList[self.vm.pages[page].section] = len(labelList)
			if len(concatLines) > 0:
				self.dataset.append(((concatLines), labelList[self.vm.pages[page].section], len(concatLines)))

			self.vocab = self.vocab.union(set(concatLines))
		print(self.labelSet)
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

def train_model(model, train_dl, epochs, lr, val_dl,lS):
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	for i in range(epochs):
		model.train()
		sum_loss = 0.0
		total = 0
		print("STARTING EPOCH "+str(i))
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
		val_loss, val_acc = validation_metrics(model, val_dl,lS)
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
lS = train.labelSet
trainD, valD = torch.utils.data.random_split(train, [127, 100])
dataloader = DataLoader(trainD, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
val_dl1 = DataLoader(valD, batch_size=1, shuffle=True, num_workers=1, drop_last=False)

l = LSTM(len(train.vocab), 300, 20)
print(validation_metrics(l, val_dl1, lS))
train_model(l, dataloader, 10, .01, val_dl1, lS)

#print(validation_metrics(l, val_dl))

# https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03
def flat_accuracy(preds, labels):
	pred_flat = np.argmax(preds, axis=1).flatten()
	labels_flat = labels.flatten()
	return np.sum(pred_flat == labels_flat) / len(labels_flat)

def Bert(input_ids, labels):
	attention_masks = []
	for seq in input_ids:
		seq_mask = [float(i>0) for i in seq]
		attention_masks.append(seq_mask)

	train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)
	train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)

	train_inputs = torch.tensor(train_inputs)
	validation_inputs = torch.tensor(validation_inputs)
	train_labels = torch.tensor(train_labels)
	validation_labels = torch.tensor(validation_labels)
	train_masks = torch.tensor(train_masks)
	validation_masks = torch.tensor(validation_masks)

	batch_size = 32

	train_data = TensorDataset(trainD, train_masks, train_labels)
	train_sampler = RandomSampler(train_data)
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
	validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
	validation_sampler = SequentialSampler(validation_data)
	validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

	model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=nb_labels)
	model.cuda()

	BertForSequenceClassification(
		(bert): BertModel(
			(embeddings): BertEmbeddings(
				(word_embeddings): Embedding(30522, 768, padding_idx=0)
				(position_embeddings): Embedding(512, 768)
				(token_type_embeddings): Embedding(2, 768)
				(LayerNorm): BertLayerNorm()
				(dropout): Dropout(p=0.1)
			)
			(encoder): BertEncoder(
				(layer): ModuleList(
					(0): BertLayer(
						(attention): BertAttention(
							(self): BertSelfAttention(
								(query): Linear(in_features=768, out_features=768, bias=True)
								(key): Linear(in_features=768, out_features=768, bias=True)
								(value): Linear(in_features=768, out_features=768, bias=True)
								(dropout): Dropout(p=0.1)
							)
							(output): BertSelfOutput(
								(dense): Linear(in_features=768, out_features=768, bias=True)
								(LayerNorm): BertLayerNorm()
								(dropout): Dropout(p=0.1)
							)
						)
						(intermediate): BertIntermediate(
							(dense): Linear(in_features=768, out_features=3072, bias=True)
						)
						(output): BertOutput(
							(dense): Linear(in_features=3072, out_features=768, bias=True)
							(LayerNorm): BertLayerNorm()
							(dropout); Dropout(p=0.1)
						)
					)
					(1): BertLayer(
						(attention): BertAttention(
							(self): BertSelfAttention(
								(query): Linear(in_features=768, out_features=768, bias=True)
								(key): Linear(in_features=768, out_features=768, bias=True)
								(value): Linear(in_features=768, out_features=768, bias=True)
								(dropout): Dropout(p=0.1)
							)
							(output): BertSelfOutput(
								(dense): Linear(in_features=768, out_features=768, bias=True)
								(LayerNorm): BertLayerNorm()
								(dropout): Dropout(p=0.1)
							)
						)
						(intermediate): BertIntermediate(
							(dense): Linear(in_features=768, out_features=3072, bias=True)
						)
						(output): BertOutput(
							(dense): Linear(in_features=3072, out_features=768, bias=True)
							(LayerNorm): BertLayerNorm()
							(dropout); Dropout(p=0.1)
						)
					)
					(2): BertLayer(
						(attention): BertAttention(
							(self): BertSelfAttention(
								(query): Linear(in_features=768, out_features=768, bias=True)
								(key): Linear(in_features=768, out_features=768, bias=True)
								(value): Linear(in_features=768, out_features=768, bias=True)
								(dropout): Dropout(p=0.1)
							)
							(output): BertSelfOutput(
								(dense): Linear(in_features=768, out_features=768, bias=True)
								(LayerNorm): BertLayerNorm()
								(dropout): Dropout(p=0.1)
							)
						)
						(intermediate): BertIntermediate(
							(dense): Linear(in_features=768, out_features=3072, bias=True)
						)
						(output): BertOutput(
							(dense): Linear(in_features=3072, out_features=768, bias=True)
							(LayerNorm): BertLayerNorm()
							(dropout); Dropout(p=0.1)
						)
					)
					(3): BertLayer(
						(attention): BertAttention(
							(self): BertSelfAttention(
								(query): Linear(in_features=768, out_features=768, bias=True)
								(key): Linear(in_features=768, out_features=768, bias=True)
								(value): Linear(in_features=768, out_features=768, bias=True)
								(dropout): Dropout(p=0.1)
							)
							(output): BertSelfOutput(
								(dense): Linear(in_features=768, out_features=768, bias=True)
								(LayerNorm): BertLayerNorm()
								(dropout): Dropout(p=0.1)
							)
						)
						(intermediate): BertIntermediate(
							(dense): Linear(in_features=768, out_features=3072, bias=True)
						)
						(output): BertOutput(
							(dense): Linear(in_features=3072, out_features=768, bias=True)
							(LayerNorm): BertLayerNorm()
							(dropout); Dropout(p=0.1)
						)
					)
					(4): BertLayer(
						(attention): BertAttention(
							(self): BertSelfAttention(
								(query): Linear(in_features=768, out_features=768, bias=True)
								(key): Linear(in_features=768, out_features=768, bias=True)
								(value): Linear(in_features=768, out_features=768, bias=True)
								(dropout): Dropout(p=0.1)
							)
							(output): BertSelfOutput(
								(dense): Linear(in_features=768, out_features=768, bias=True)
								(LayerNorm): BertLayerNorm()
								(dropout): Dropout(p=0.1)
							)
						)
						(intermediate): BertIntermediate(
							(dense): Linear(in_features=768, out_features=3072, bias=True)
						)
						(output): BertOutput(
							(dense): Linear(in_features=3072, out_features=768, bias=True)
							(LayerNorm): BertLayerNorm()
							(dropout); Dropout(p=0.1)
						)
					)
					(5): BertLayer(
						(attention): BertAttention(
							(self): BertSelfAttention(
								(query): Linear(in_features=768, out_features=768, bias=True)
								(key): Linear(in_features=768, out_features=768, bias=True)
								(value): Linear(in_features=768, out_features=768, bias=True)
								(dropout): Dropout(p=0.1)
							)
							(output): BertSelfOutput(
								(dense): Linear(in_features=768, out_features=768, bias=True)
								(LayerNorm): BertLayerNorm()
								(dropout): Dropout(p=0.1)
							)
						)
						(intermediate): BertIntermediate(
							(dense): Linear(in_features=768, out_features=3072, bias=True)
						)
						(output): BertOutput(
							(dense): Linear(in_features=3072, out_features=768, bias=True)
							(LayerNorm): BertLayerNorm()
							(dropout); Dropout(p=0.1)
						)
					)
					(6): BertLayer(
						(attention): BertAttention(
							(self): BertSelfAttention(
								(query): Linear(in_features=768, out_features=768, bias=True)
								(key): Linear(in_features=768, out_features=768, bias=True)
								(value): Linear(in_features=768, out_features=768, bias=True)
								(dropout): Dropout(p=0.1)
							)
							(output): BertSelfOutput(
								(dense): Linear(in_features=768, out_features=768, bias=True)
								(LayerNorm): BertLayerNorm()
								(dropout): Dropout(p=0.1)
							)
						)
						(intermediate): BertIntermediate(
							(dense): Linear(in_features=768, out_features=3072, bias=True)
						)
						(output): BertOutput(
							(dense): Linear(in_features=3072, out_features=768, bias=True)
							(LayerNorm): BertLayerNorm()
							(dropout); Dropout(p=0.1)
						)
					)
					(7): BertLayer(
						(attention): BertAttention(
							(self): BertSelfAttention(
								(query): Linear(in_features=768, out_features=768, bias=True)
								(key): Linear(in_features=768, out_features=768, bias=True)
								(value): Linear(in_features=768, out_features=768, bias=True)
								(dropout): Dropout(p=0.1)
							)
							(output): BertSelfOutput(
								(dense): Linear(in_features=768, out_features=768, bias=True)
								(LayerNorm): BertLayerNorm()
								(dropout): Dropout(p=0.1)
							)
						)
						(intermediate): BertIntermediate(
							(dense): Linear(in_features=768, out_features=3072, bias=True)
						)
						(output): BertOutput(
							(dense): Linear(in_features=3072, out_features=768, bias=True)
							(LayerNorm): BertLayerNorm()
							(dropout); Dropout(p=0.1)
						)
					)
					(8): BertLayer(
						(attention): BertAttention(
							(self): BertSelfAttention(
								(query): Linear(in_features=768, out_features=768, bias=True)
								(key): Linear(in_features=768, out_features=768, bias=True)
								(value): Linear(in_features=768, out_features=768, bias=True)
								(dropout): Dropout(p=0.1)
							)
							(output): BertSelfOutput(
								(dense): Linear(in_features=768, out_features=768, bias=True)
								(LayerNorm): BertLayerNorm()
								(dropout): Dropout(p=0.1)
							)
						)
						(intermediate): BertIntermediate(
							(dense): Linear(in_features=768, out_features=3072, bias=True)
						)
						(output): BertOutput(
							(dense): Linear(in_features=3072, out_features=768, bias=True)
							(LayerNorm): BertLayerNorm()
							(dropout); Dropout(p=0.1)
						)
					)
					(9): BertLayer(
						(attention): BertAttention(
							(self): BertSelfAttention(
								(query): Linear(in_features=768, out_features=768, bias=True)
								(key): Linear(in_features=768, out_features=768, bias=True)
								(value): Linear(in_features=768, out_features=768, bias=True)
								(dropout): Dropout(p=0.1)
							)
							(output): BertSelfOutput(
								(dense): Linear(in_features=768, out_features=768, bias=True)
								(LayerNorm): BertLayerNorm()
								(dropout): Dropout(p=0.1)
							)
						)
						(intermediate): BertIntermediate(
							(dense): Linear(in_features=768, out_features=3072, bias=True)
						)
						(output): BertOutput(
							(dense): Linear(in_features=3072, out_features=768, bias=True)
							(LayerNorm): BertLayerNorm()
							(dropout); Dropout(p=0.1)
						)
					)
					(10): BertLayer(
						(attention): BertAttention(
							(self): BertSelfAttention(
								(query): Linear(in_features=768, out_features=768, bias=True)
								(key): Linear(in_features=768, out_features=768, bias=True)
								(value): Linear(in_features=768, out_features=768, bias=True)
								(dropout): Dropout(p=0.1)
							)
							(output): BertSelfOutput(
								(dense): Linear(in_features=768, out_features=768, bias=True)
								(LayerNorm): BertLayerNorm()
								(dropout): Dropout(p=0.1)
							)
						)
						(intermediate): BertIntermediate(
							(dense): Linear(in_features=768, out_features=3072, bias=True)
						)
						(output): BertOutput(
							(dense): Linear(in_features=3072, out_features=768, bias=True)
							(LayerNorm): BertLayerNorm()
							(dropout); Dropout(p=0.1)
						)
					)
					(11): BertLayer(
						(attention): BertAttention(
							(self): BertSelfAttention(
								(query): Linear(in_features=768, out_features=768, bias=True)
								(key): Linear(in_features=768, out_features=768, bias=True)
								(value): Linear(in_features=768, out_features=768, bias=True)
								(dropout): Dropout(p=0.1)
							)
							(output): BertSelfOutput(
								(dense): Linear(in_features=768, out_features=768, bias=True)
								(LayerNorm): BertLayerNorm()
								(dropout): Dropout(p=0.1)
							)
						)
						(intermediate): BertIntermediate(
							(dense): Linear(in_features=768, out_features=3072, bias=True)
						)
						(output): BertOutput(
							(dense): Linear(in_features=3072, out_features=768, bias=True)
							(LayerNorm): BertLayerNorm()
							(dropout); Dropout(p=0.1)
						)
					)
				)
			)
			(pooler): BertPooler(
				(dense): Linear(in_features=768, out_features=768, bias=True)
				(activation): Tanh()
			)
		)
		(dropout): Dropout(p=0.1)
		(classifier): Linear(in_features=768, out_features=2, bias=True)
	)

	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'gamma', 'beta']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
		 'weight_decay_rate': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
		 'weight_decay_rate': 0.0}
	]

	optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=0.1)

	train_loss_set = []
	epochs = 4
	for _ in trange(epochs, desc="Epoch"):
		model.train()

		tr_loss = 0
		nb_tr_examples, nb_tr_steps = 0, 0

		for step, batch in enumerate(train_dataloader):

			batch = tuple(t.to(device) for t in batch)

			b_input_ids, b_input_mask, b_labels = batch

			optimizer.zero_grad()

			loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
			train_loss_set.append(loss.item())

			loss.backward()

			optimizer.step()

			tr_loss += loss.item()
			nb_tr_examples += b_input_ids.size(0)
			nb_tr_steps += 1
		print("Train loss: {}".format(tr_loss/nb_tr_steps))

		model.eval()

		eval_loss, eval_accuracy = 0, 0
		nb_eval_steps, nb_eval_examples = 0, 0

		for batch in validation_dataloader:

			batch = tuple(t.to(device) for t in batch)

			b_input_ids, b_input_mask, b_labels = batch

			with torch.no_grad():
				logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

			logits = logits.detach().cpu().numpy()
			label_ids = b_labels.to('cpu').numpy()
			tmp_eval_accuracy = flat_accuracy(logits, label_ids)
			eval_accuracy += tmp_eval_accuracy
			nb_eval_steps += 1
		print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))






