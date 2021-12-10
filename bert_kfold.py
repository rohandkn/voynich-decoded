# -*- coding: utf-8 -*-
"""Untitled9.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1gtKBv9Ggvugl2usf03Msd2MfC8Zbg3AT
"""

#!pip install pytorch_pretrained_bert

from voynich import VoynichManuscript
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torchtext.legacy.data import Field, TabularDataset, BucketIterator 
from sklearn.metrics import classification_report
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from sklearn.model_selection import train_test_split
from tqdm import trange
from sklearn.model_selection import KFold

def flat_accuracy(preds, labels):
  pred_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)

def shap_get_sum(line_count, fold):
  shap_values_data = np.load("shap_values_data" + str(fold) + ".npy", allow_pickle=True)
  shap_values_values = np.load("shap_values_values" + str(fold) + ".npy", allow_pickle=True)
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
  np.save("shap_sum" + str(fold) + ".npy", shap_total_vals)
  print("Saved to " + "shap_sum" + str(fold) + ".npy")

def shap_get_max(line_count, fold):
  shap_values_data = np.load("shap_values_data" + str(fold) + ".npy", allow_pickle=True)
  shap_values_values = np.load("shap_values_values" + str(fold) + ".npy", allow_pickle=True)
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
  np.save("shap_max" + str(fold) + ".npy", shap_total_vals)
  print("Saved to " + "shap_max" + str(fold) + ".npy")

def Bert():
  kcount = 1
  kf = KFold(n_splits=10, shuffle=True)
  val_accs = []
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

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  tokenized_texts = [tokenizer.tokenize(line) for line in lines]

  MAX_LEN = 128
  input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                            maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
  input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
  input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")


  attention_masks = []
  for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)
  


  model = BertForSequenceClassification.from_pretrained("bestmodel.rpt", num_labels=6)
  model.to(device)

  param_optimizer = list(model.named_parameters())
  no_decay = ['bias', 'gamma', 'beta']
  optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
      'weight_decay_rate': 0.01},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
      'weight_decay_rate': 0.0}
    ]

  optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=0.1)

  for train_index, test_index in kf.split(input_ids, labels):
    #print(len(input_ids))
    labels = np.array(labels)
    attention_masks = np.array(attention_masks)
    print("k-fold:", kcount)
    kcount += 1
    train_inputs, validation_inputs = input_ids[train_index], input_ids[test_index]
    train_labels, validation_labels = labels[train_index], labels[test_index]
    train_masks, validation_masks = attention_masks[train_index], attention_masks[test_index]
     

    #train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)
    #train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)

    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    batch_size = 32

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    train_loss_set = []
    epochs = 0
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
    torch.save(model.state_dict(), "bestmodel.rpt")
  #model.to('cpu')
  pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
  prediction = pipe(lines)
  explainer = shap.Explainer(pipe)
  shap_values = explainer(lines)
  np.save("shap_values_values.npy", shap_values.values)
  np.save("shap_values_data.npy", shap_values.data)
  return val_accs

valacc = Bert()

print(sum(valacc)/len(valacc))


#shap_get_sum(54, 2)
#shap_get_max(54, 2)

#shap_get_sum(54, 3)
#shap_get_max(54, 3)

#shap_get_sum(54, 4)
#shap_get_max(54, 4)

#print("Num Lines: " + str(len(lines[::100])))