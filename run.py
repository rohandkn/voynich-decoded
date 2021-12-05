from voynich import VoynichManuscript
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.legacy.data import Field, TabularDataset, BucketIterator	
from sklearn.metrics import classification_report
import numpy as np


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






