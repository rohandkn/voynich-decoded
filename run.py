from voynich import VoynichManuscript
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torchtext.legacy.data import Field, TabularDataset, BucketIterator	
from sklearn.metrics import classification_report
import numpy as np
from transformers import *
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import trange
import shap

def shap_get_sum(line_count):
  shap_values_data = np.load("shap_values_data.npy", allow_pickle=True)
  shap_values_values = np.load("shap_values_values.npy", allow_pickle=True)
  shap_total_vals = np.array([{}, {}, {}, {}, {}, {}, {}])
  for i in range(line_count):
    curr_string = ""
    shap_weights = np.zeros(6)
    for j in range(len(shap_values_data[i])):
      if shap_values_data[i][j].strip(' ').isalpha():
        if shap_values_data[i][j][0] == ' ':
          if len(curr_string) > 0:
            for k in range(6):
              if curr_string in shap_total_vals[k]:
                shap_total_vals[k][curr_string] += np.abs(shap_weights[k])
                if k == 0:
                  shap_total_vals[6][curr_string] += 1
              else:
                shap_total_vals[k][curr_string] = np.abs(shap_weights[k])
                if k == 0:
                  shap_total_vals[6][curr_string] = 1
          curr_string = shap_values_data[i][j][1:]
          shap_weights = np.abs(shap_values_values[i][j])
        else:
          curr_string += shap_values_data[i][j]
          shap_weights += np.abs(shap_values_values[i][j])
      else:
        if len(curr_string) > 0:
          for k in range(6):
            if curr_string in shap_total_vals[k]:
              shap_total_vals[k][curr_string] += np.abs(shap_weights[k])
              if k == 0:
                shap_total_vals[6][curr_string] += 1
            else:
              shap_total_vals[k][curr_string] = np.abs(shap_weights[k])
              if k == 0:
                shap_total_vals[6][curr_string] = 1
        curr_string = ""
        shap_weights = np.zeros(6)
  np.save("shap_sum.npy", shap_total_vals)
  print("Saved to " + "shap_sum.npy")

def shap_get_max(line_count):
  shap_values_data = np.load("shap_values_data.npy", allow_pickle=True)
  shap_values_values = np.load("shap_values_values.npy", allow_pickle=True)
  shap_total_vals = np.array([{}, {}, {}, {}, {}, {}])
  for i in range(line_count):
    curr_string = ""
    shap_weights = np.zeros(6)
    for j in range(len(shap_values_data[i])):
      if shap_values_data[i][j].strip(' ').isalpha():
        if shap_values_data[i][j][0] == ' ':
          if len(curr_string) > 0:
            for k in range(6):
              if curr_string in shap_total_vals[k]:
                if np.abs(shap_weights[k]) > shap_total_vals[k][curr_string]:
                  shap_total_vals[k][curr_string] = np.abs(shap_weights[k])
              else:
                shap_total_vals[k][curr_string] = np.abs(shap_weights[k])
          curr_string = shap_values_data[i][j][1:]
          shap_weights = np.abs(shap_values_values[i][j])
        else:
          curr_string += shap_values_data[i][j]
          shap_weights += np.abs(shap_values_values[i][j])
      else:
        if len(curr_string) > 0:
          for k in range(6):
            if curr_string in shap_total_vals[k]:
              if np.abs(shap_weights[k]) > shap_total_vals[k][curr_string]:
                shap_total_vals[k][curr_string] = np.abs(shap_weights[k])
            else:
              shap_total_vals[k][curr_string] = np.abs(shap_weights[k])
        curr_string = ""
        shap_weights = np.zeros(6)
  np.save("shap_max.npy", shap_total_vals)
  print("Saved to " + "shap_max.npy")

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

#vm = VoynichManuscript("voynich-text.txt", inline_comments=False)
#train = VoynichDataset()
#lS = train.labelSet
#trainD, valD = torch.utils.data.random_split(train, [127, 100])
#dataloader = DataLoader(trainD, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
#val_dl1 = DataLoader(valD, batch_size=1, shuffle=True, num_workers=1, drop_last=False)

#l = LSTM(len(train.vocab), 300, 20)
#print(validation_metrics(l, val_dl1, lS))
#train_model(l, dataloader, 10, .01, val_dl1, lS)

#print(validation_metrics(l, val_dl))

# https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03
def flat_accuracy(preds, labels):
	pred_flat = np.argmax(preds, axis=1).flatten()
	labels_flat = labels.flatten()
	return np.sum(pred_flat == labels_flat) / len(labels_flat)

def flat_acc_eval(x):
    return {"flat":flat_accuracy(x.predictions, x.label_ids)}


def Bert():
	lines = []
	labelNums = {}
	labelCount = 0
	labels = []
	vm = VoynichManuscript("voynich-text.txt", inline_comments=False)
	for page in vm.pages:
		if vm.pages[page].section in labelNums:
			section_label = labelNums[vm.pages[page].section]
		else:
			section_label = labelCount
			labelNums[vm.pages[page].section] = labelCount
			labelCount += 1
		for line in vm.pages[page]:
			lines.append("[CLS]" + line.text.replace(".", " ") + "[SEP]")
			labels.append(section_label)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	n_gpu = torch.cuda.device_count()
	#torch.cuda.get_device_name(0)

	tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', do_lower_case=True)

	def encode_with_truncation(examples):


	  """Mapping function to tokenize the sentences passed with truncation"""

	  print(type(examples))
	  return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

	#dataset = load_dataset("csv", delimiter='/', data_files=["fullTrain.csv"], split="train")
	
	#d = dataset.train_test_split(test_size=0.05)


	#train_dataset = d["train"].map(encode_with_truncation, batched=True)
	# tokenizing the testing dataset
	#test_dataset = d["test"].map(encode_with_truncation, batched=True)



	model = XLMRobertaForSequenceClassification.from_pretrained("hf-model/checkpoint-100", num_labels=6)
	model.to(device)

	#trainer.train()
	print(len(lines[::3]))
	exit(0)

	model.to('cpu')
	pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
	#prediction = pipe(lines)
	explainer = shap.Explainer(pipe)
	shap_values = explainer(lines[::3])
	np.save("shap_values_values.npy", shap_values.values)
	np.save("shap_values_data.npy", shap_values.data)

Bert()






