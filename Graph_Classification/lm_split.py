import os
import re
import random


files = []
directory = './mdivi_llo'

for filename in os.listdir(directory):
	files.append(filename)


random.shuffle(files)
#print(files)

test_lm = []
train_lm = []
validation_lm = []


def label(filename):
    if re.search("LLO", filename):
        return [1]

    else:
        return [0]

llo = 0
mdivi = 0
control = 0
 
for f in files: 

	source_folder = directory + '/' + f
	#print(source_folder)
	
	target = label(f)
	if(target == [1] and llo <= 18):
		train_lm.append(f)
		llo = llo+1

	#elif(target == [1] and llo>18 and llo <=22):
	#	validation_lm.append(f)
	#	llo = llo+1
	elif(target == [1] and llo > 18 and llo <=30):
		test_lm.append(f)
		llo = llo+1
	
	if(target == [0] and mdivi <= 18):
		train_lm.append(f)
		mdivi = mdivi+1
	#elif(target == [0] and mdivi>18 and mdivi <=22):
	#	validation_lm.append(f)
	#	mdivi = mdivi+1
	elif(target == [0] and mdivi >18): #>=22):
		test_lm.append(f)
		mdivi = mdivi+1
	'''
	if(target == [0,1] and control <= 18):
		train_lm.append(f)
		control = control+1
	#elif(target == [0,1] and control>18 and control <=22):
	#	validation_lm.append(f)
	#	control = control+1
	elif(target == [0,1] and control > 18):
		test_lm.append(f)
		control = control+1
	'''


print('train_lm',len(train_lm))
print('test_lm', len(test_lm))
print('validation_lm', len(validation_lm))
'''
print(train_lm)
print(test_lm)
print(validation_lm)


for f in test_lm: 
	print(f, ": ", label(f))

'''