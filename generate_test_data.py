

from voynich import VoynichManuscript

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from keras.preprocessing.sequence import pad_sequences

import random
import numpy as np



def random_pair_generator(numbers): 
    seen = set() 
    
    while True: 
        pair = random.sample(numbers, 2) 
        pair = tuple(sorted(pair)) 
        if pair not in seen: 
            seen.add(pair) 
            yield pair


# generate pairs of inputs for test data
# three files are generated, each containing a torch tensor
# inputs file: contains pair of input tokens
# masks file: contains pair of attention masks
# labels file: contains labels for each pair
#
# for the inputs and mask files:
#   the pair index is along axis 0
#   axis 1 is of length 2, with each element representing a different input
#
# for the labels file:
#   the pair index is along axis 0
#   axis 1 is of length 1, the one element represents the label for the pair
def generate_test_data(inputs_filename, masks_filename, labels_filename):
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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(line) for line in lines]

    MAX_LEN = 128
    # input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
    #                         maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")


    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)


    # generate text comparison pairs
    NUM_TEST_PAIRS = 1000
    input_indices = list(range(len(input_ids)))
    pair_gen = random_pair_generator(input_indices)

    test_inputs = []
    test_masks = []
    test_labels = []
    for n in range(NUM_TEST_PAIRS):
        pair = next(pair_gen)
        i = pair[0]
        j = pair[1]
        test_inputs.append([input_ids[i], input_ids[j]])
        test_masks.append([attention_masks[i], attention_masks[j]])
        test_label = 1 if labels[i] == labels[j] else 0
        test_labels.append(test_label)

    test_inputs = torch.tensor(test_inputs)
    test_masks = torch.tensor(test_masks)
    test_labels = torch.tensor(test_labels)

    torch.save(test_inputs, inputs_filename)
    torch.save(test_masks, masks_filename)
    torch.save(test_labels, labels_filename)


def load_test_data(inputs_filename, masks_filename, labels_filename):
    test_inputs = torch.load(inputs_filename)
    test_masks = torch.load(masks_filename)
    test_labels = torch.load(labels_filename)
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    return test_data



if __name__ == "__main__":
    generate_test_data("test_inputs.pt", "test_masks.pt", "test_labels.pt")