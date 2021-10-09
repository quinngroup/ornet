import os
import re

llo = []
mdivi = []
control = []

directory = './mdivi_llo'

for filename in os.listdir(directory):
    if re.search("Mdivi", filename):
        mdivi.append(filename)
    elif re.search("LLO", filename):
        llo.append(filename)
    else:
        control.append(filename)
print('llo',len(llo))
print('mdivi',len(mdivi))
print('control',len(control))