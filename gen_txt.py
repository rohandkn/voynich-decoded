from voynich import VoynichManuscript
import random


labelCount = 0
labels = []
lines = []
labelNums = {}

vm = VoynichManuscript("voynich-text.txt", inline_comments=False)
for page in vm.pages:
	if vm.pages[page].section in labelNums:
		section_label = labelNums[vm.pages[page].section]
	else:
		section_label = labelCount
		labelNums[vm.pages[page].section] = labelCount
		labelCount += 1
	for line in vm.pages[page]:
		lines.append(line.text.replace(".", " "))
		labels.append(section_label)


with open('full.txt', 'a') as full:
	for line in lines:
		train.write(line+"\n")