import json

with open('testlist01.txt','r') as f:
	test_lines = f.readlines()

for i,l in enumerate(test_lines):
	test_lines[i] = l.replace('\n','').replace('.avi','')

with open('trainlist01.txt','r') as f:
	train_lines = f.readlines()

for i,l in enumerate(train_lines):
	train_lines[i] = l.split()[0].replace('\n','').replace('.avi','')

with open('test.json','w') as f:
	json.dump(test_lines, f)


with open('train.json','w') as f:
	json.dump(train_lines, f)