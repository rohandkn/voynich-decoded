from voynich import VoynichManuscript
import random
import string
import re

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
            pline = re.sub(r'[^A-Za-z0-9.]+', '', line.text)
            pline = pline.replace(".", " ")
            print(type(pline))
            lines.append(pline)
            labels.append(section_label)


with open('full.txt','w') as full:
	for line in lines:
		full.write(line+"\n")

with open('fullTrain.csv', 'w') as full:
        full.write("text/label\n")
        for i in range(0, len(lines)):
            full.write(lines[i]+"/"+str(labels[i])+"\n")
