import os
import re
import random


files = []
directory = './llo_control'

for filename in os.listdir(directory):
	files.append(filename)


random.shuffle(files)
#print(files)

test = []
train = []
validation = []


def label(filename):
    if re.search("LLO", filename):
        return [0]

    else:
        return [1]

llo = 0
mdivi = 0
control = 0
 
for f in files: 

	source_folder = directory + '/' + f
	#print(source_folder)
	
	target = label(f)
	if(target == [0] and llo <= 18):
		train.append(f)
		llo = llo+1
	elif(target == [0] and llo>18 and llo <=22):
		validation.append(f)
		llo = llo+1
	elif(target == [0] and llo >= 22 and llo <=30):
		test.append(f)
		llo = llo+1
	'''
	if(target == [1,0,0] and mdivi <= 18):
		train.append(f)
		mdivi = mdivi+1
	elif(target == [1,0,0] and mdivi>18 and mdivi <=22):
		validation.append(f)
		mdivi = mdivi+1
	elif(target == [1,0,0] and mdivi >= 22):
		test.append(f)
		mdivi = mdivi+1
	'''
	if(target == [1] and control <= 18):
		train.append(f)
		control = control+1
	elif(target == [1] and control>18 and control <=22):
		validation.append(f)
		control = control+1
	elif(target == [1] and control >= 22):
		test.append(f)
		control = control+1


'''
print('train',len(train))
print('test', len(test))
print('validation', len(validation))

print(train)
print(test)
print(validation)
'''