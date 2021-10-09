import subprocess
import sys
import matplotlib.pyplot as plt
acc = []
a_file = open("run_binary_accuracy.txt")
lines = a_file.readlines()
for l in lines: 
	acc.append(float(l[-8:-4]))

x= list(range(1,101))

'''
k = 0.6

count = 0
for i in acc :
    if i > k :
        count = count + 1



print('greater than 50: ', count)
'''



fig = plt.hist(acc, 40)
  
# naming the x axis
plt.xlabel('time ')
# naming the y axis
plt.ylabel('accuracy')
  
# giving a title to my graph
plt.title('Run 4')
  
# function to show the plot
plt.show()

