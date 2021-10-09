import os
import re
import random


files = []
directory = './mdivi_control'

for filename in os.listdir(directory):
	files.append(filename)


random.shuffle(files)
#print(files)

test_mc = []
train_mc = []
validation_mc = []


def label(filename):
    if re.search("Mdivi", filename):
        return [0]

    else:
        return [1]
llo = 0
mdivi = 0
control = 0

for f in files: 

	target = label(f)
	source_folder = directory + '/' + f
	#print(source_folder)
	'''
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
	'''
	if(target == [0] and mdivi <= 18):
		train_mc.append(f)
		mdivi = mdivi+1
	#elif(target == [0] and mdivi>18 and mdivi <=22):
	#	validation_lm.append(f)
	#	mdivi = mdivi+1
	elif(target == [0] and mdivi >18): #>=22):
		test_mc.append(f)
		mdivi = mdivi+1


	if(target == [1] and control <= 18):
		train_mc.append(f)
		control = control+1
	#elif(target == [1] and control>18 and control <=22):
	#	validation.append(f)
	#	control = control+1
	elif(target == [1] and control > 18):
		test_mc.append(f)
		control = control+1
