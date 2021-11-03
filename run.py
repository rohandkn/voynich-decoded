from voynich import VoynichManuscript
from torch.utils.data import Dataset, DataLoader
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import torch


	

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



train = VoynichDataset()


