
import os
import subprocess

graph_dir = '/extrastorage/ornet/distances/Hellinger Distances/distances'
args = ['python', '-m', 'ornet.eigenspectrum', '-o', '/home/marcus/Desktop/Plots/']
for filename in os.listdir(graph_dir):
     arg_copy = args.copy()
     arg_copy.append('-i')
     arg_copy.append(os.path.join(graph_dir, filename))
     subprocess.run(arg_copy)
