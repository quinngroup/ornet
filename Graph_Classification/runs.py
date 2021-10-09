import subprocess
import sys

original_stdout = sys.stdout
with open('run_binary_accuracy.txt', 'w') as f:
	sys.stdout = f
	for i in range(100):
	
		out = subprocess.check_output(['python3' , 'mdivi_vs_llo.py'])
		print(out)
	

	sys.stdout = original_stdout