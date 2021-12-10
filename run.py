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

	dataset = load_dataset("csv", delimiter='/', data_files=["fullTrain.csv"], split="train")
	
	d = dataset.train_test_split(test_size=0.05)


	train_dataset = d["train"].map(encode_with_truncation, batched=True)
	# tokenizing the testing dataset
	test_dataset = d["test"].map(encode_with_truncation, batched=True)



	model = XLMRobertaForSequenceClassification.from_pretrained("hf-model/checkpoint-100", num_labels=6)
	model.to(device)

	training_args = TrainingArguments(
            output_dir="hf-model",          # output directory to where save model checkpoint
	    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
	    overwrite_output_dir=True,      
	    num_train_epochs=100,            # number of training epochs, feel free to tweak
	    per_device_train_batch_size=3, # the training batch size, put it as high as your GPU memory fits
	    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
	    per_device_eval_batch_size=3,  # evaluation batch size
	    logging_steps=100,             # evaluate, log and save model checkpoints every 1000 step
	    save_steps=100,
	    warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight
            # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
	    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
	)

	# initialize the trainer and pass everything to it
	trainer = Trainer(
	    model=model,
            compute_metrics=flat_acc_eval,
	    args=training_args,
	    train_dataset=train_dataset,
	    eval_dataset=test_dataset,
	)

	#trainer.train()

	model.to('cpu')
	pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
	#prediction = pipe(lines)
	explainer = shap.Explainer(pipe[::3])
	shap_values = explainer(lines)
	np.save("shap_values_values.npy", shap_values.values)
	np.save("shap_values_data.npy", shap_values.data)

Bert()




